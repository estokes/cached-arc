use std::{
    borrow, fmt, mem, isize, usize,
    sync::atomic::{self, Ordering::{Acquire, Relaxed, Release, SeqCst}},
    cmp::Ordering,
    ops::{Deref, DerefMut},
    ptr::{self, NonNull},
    marker::{Unpin, PhantomData},
    hash::{Hash, Hasher},    
    alloc::{System, GlobalAlloc, Layout},
    boxed::Box,
    collections::HashMap,
    cell::RefCell,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Discriminant(usize);

pub unsafe trait Cacheable {
    /// return the type to the state it was when it was just
    /// allocated. e.g. Vec::clear, or similar. It is, of course, not
    /// necessary to deallocate any memory, as long as it is cleared.
    /// reinit must not clone, or otherwise store any references to
    /// the cached value!
    fn reinit(&mut self);

    /// return the type discriminant that you want to use to classify
    /// the type. Each type need not have a unique discriminant, but
    /// types that share a disciminant must be isomorphic, including
    /// with respect to drop.
    ///
    /// for example it would be safe to return the same discriminant
    /// for, forall<T>: Vec<T> where size_of::<T>() == usize, and
    /// reinit() cleares the Vec, because,
    ///
    /// 1. the size of the Vec allocation will is the same for all Vec<T>
    /// 
    /// 2. the Vec is guaranteed to be empty when it is in the cache,
    /// so no T ever needs to be dropped.
    fn type_id() -> Discriminant;

    /// how many arcs of this type should we cache?
    fn limit() -> usize;
}

struct PoolByType {
    // vals is a vec of pointers to the malloced arcs. we rely on the
    // typeid to ensure we return an isomorphic structure when we
    // return an element from the cache and cast it to the requested
    // type.
    vals: Vec<usize>,
    drop: Box<dyn Fn(usize) -> ()>,
}

struct Pool(HashMap<Discriminant, PoolByType>);

impl Deref for Pool {
    type Target = HashMap<Discriminant, PoolByType>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Pool {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl Drop for Pool {
    // the thread local pool will be dropped whenever a thread dies,
    // so we need to make sure drop is called on all the cached values.
    fn drop(&mut self) {
        for p in self.values_mut() {
            do_clear_cache_for(p);
        }
    }
}

thread_local! {
    static POOL: RefCell<Pool> = RefCell::new(Pool(HashMap::new()));
    static NEXT_DISCRIMINANT: RefCell<usize> = RefCell::new(0);
}

/// gets a new discriminant which is unique on this thread. You must
/// store it in thread local storage and reuse it for each isomophic
/// type.
pub fn new_discriminant() -> Discriminant {
    NEXT_DISCRIMINANT.with(|r| {
        let mut r = r.borrow_mut();
        let v = *r;
        *r += 1;
        Discriminant(v)
    })
}

fn do_clear_cache_for(pool: &mut PoolByType) {
    pool.vals.drain(0..).for_each(&*pool.drop)
}

/// Drop all cached values of the specified discriminant cached by
/// this thread.
pub fn clear_cache_for(typ: Discriminant) {
    POOL.with(|inner| {
        let mut inner = inner.borrow_mut();
        if let Some(pool) = inner.get_mut(&typ) {
            do_clear_cache_for(pool);
        }
    })
}

/// Drop all the values of all discriminants cached by this thread
pub fn clear_cache() {
    POOL.with(|p| p.borrow().iter().map(|(k, _)| *k).collect::<Vec<_>>())
        .into_iter().for_each(|d| clear_cache_for(d));
}

// Take a value of type T from the cache if one is available.
fn take<T: Cacheable>() -> Option<Arc<T>> {
    let tid = T::type_id();
    POOL.with(|inner| {
        let mut inner = inner.borrow_mut();
        inner.get_mut(&tid).and_then(|p| p.vals.pop().map(|ptr| unsafe {
            let t = mem::transmute::<usize, NonNull<ArcInner<T>>>(ptr);
            let t = Arc::from_inner(t);
            t.inner().strong.fetch_add(1, Relaxed);
            t
        }))
    })
}

// try to add an arc to the cache, return true if it was added, false
// if it wasn't.
fn try_put<T: Cacheable>(v: &mut Arc<T>) -> bool {
    let tid = T::type_id();
    POOL.with(|inner| {
        let mut inner = inner.borrow_mut();
        let pool = inner.entry(tid).or_insert_with(|| PoolByType {
            vals: Vec::new(),
            drop: Box::new(|ptr| unsafe {
                let t = mem::transmute::<usize, NonNull<ArcInner<T>>>(ptr);
                let mut t = Arc::from_inner(t);
                t.drop_slow();
            }),
        });
        if pool.vals.len() < T::limit() {
            unsafe { T::reinit(Arc::get_mut_unchecked(v)); }
            let t = unsafe { mem::transmute::<NonNull<ArcInner<T>>, usize>(v.ptr) };
            for v in &pool.vals {
                assert!(v != &t);
            }
            pool.vals.push(t);
            true
        } else {
            false
        }
    })
}

const MAX_REFCOUNT: usize = (isize::MAX) as usize;

fn is_dangling<T: ?Sized>(ptr: NonNull<T>) -> bool {
    let address = ptr.as_ptr() as *mut () as usize;
    address == usize::MAX
}

pub struct Arc<T: Cacheable> {
    ptr: NonNull<ArcInner<T>>,
    phantom: PhantomData<T>,
}

unsafe impl<T: Sync + Send + Cacheable> Send for Arc<T> {}
unsafe impl<T: Sync + Send + Cacheable> Sync for Arc<T> {}

impl<T: Cacheable> Arc<T> {
    fn from_inner(ptr: NonNull<ArcInner<T>>) -> Self {
        Self {
            ptr,
            phantom: PhantomData,
        }
    }
}

pub struct Weak<T: Cacheable> {
    // This is a `NonNull` to allow optimizing the size of this type in enums,
    // but it is not necessarily a valid pointer.
    // `Weak::new` sets this to `usize::MAX` so that it doesn’t need
    // to allocate space on the heap.  That's not a value a real pointer
    // will ever have because RcBox has alignment at least 2.
    ptr: NonNull<ArcInner<T>>,
}

unsafe impl<T: Sync + Send + Cacheable> Send for Weak<T> {}
unsafe impl<T: Sync + Send + Cacheable> Sync for Weak<T> {}

impl<T: fmt::Debug + Cacheable> fmt::Debug for Weak<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(Weak)")
    }
}

struct ArcInner<T: Cacheable> {
    strong: atomic::AtomicUsize,
    // the value usize::MAX acts as a sentinel for temporarily "locking" the
    // ability to upgrade weak pointers or downgrade strong ones; this is used
    // to avoid races in `make_mut` and `get_mut`.
    weak: atomic::AtomicUsize,
    data: T,
}

unsafe impl<T: Sync + Send + Cacheable> Send for ArcInner<T> {}
unsafe impl<T: Sync + Send + Cacheable> Sync for ArcInner<T> {}

impl<T: Cacheable> Arc<T> {
    /// Force construction of a new Arc without looking in the cache
    /// for an existing one that can be reused.
    #[inline]
    pub fn new_no_cache(data: T) -> Arc<T> {
        // Start the weak pointer count as 1 which is the weak pointer that's
        // held by all the strong pointers (kinda), see std/rc.rs for more info
        let x: Box<_> = Box::new(ArcInner {
            strong: atomic::AtomicUsize::new(1),
            weak: atomic::AtomicUsize::new(1),
            data,
        });
        Self::from_inner(NonNull::new(Box::into_raw(x)).unwrap())
    }

    /// Look in the cache for an Arc of type Arc<T>, otherwise
    /// construct a new one with the result of calling f.
    pub fn new<F: FnOnce() -> T>(f: F) -> Arc<T> {
        take::<T>().unwrap_or_else(|| Arc::new_no_cache(f()))
    }

    // CR estokes: Is this safe with the cache? Does it trigger a drop?
    #[inline]
    pub fn try_unwrap(this: Self) -> Result<T, Self> {
        // See `drop` for why all these atomics are like this
        if this.inner().strong.compare_exchange(1, 0, Release, Relaxed).is_err() {
            return Err(this);
        }

        atomic::fence(Acquire);

        unsafe {
            let elem = ptr::read(&this.ptr.as_ref().data);

            // Make a weak pointer to clean up the implicit strong-weak reference
            let _weak = Weak { ptr: this.ptr };
            mem::forget(this);

            Ok(elem)
        }
    }
}

impl<T: Cacheable> Arc<T> {
    pub fn into_raw(this: Self) -> *const T {
        let ptr: *const T = &*this;
        mem::forget(this);
        ptr
    }

    pub fn downgrade(this: &Self) -> Weak<T> {
        // This Relaxed is OK because we're checking the value in the CAS
        // below.
        let mut cur = this.inner().weak.load(Relaxed);

        loop {
            // check if the weak counter is currently "locked"; if so, spin.
            if cur == usize::MAX {
                cur = this.inner().weak.load(Relaxed);
                continue;
            }

            // NOTE: this code currently ignores the possibility of overflow
            // into usize::MAX; in general both Rc and Arc need to be adjusted
            // to deal with overflow.

            // Unlike with Clone(), we need this to be an Acquire read to
            // synchronize with the write coming from `is_unique`, so that the
            // events prior to that write happen before this read.
            match this.inner().weak.compare_exchange_weak(cur, cur + 1, Acquire, Relaxed) {
                Ok(_) => {
                    // Make sure we do not create a dangling Weak
                    debug_assert!(!is_dangling(this.ptr));
                    return Weak { ptr: this.ptr };
                }
                Err(old) => cur = old,
            }
        }
    }

    #[inline]
    pub fn weak_count(this: &Self) -> usize {
        let cnt = this.inner().weak.load(SeqCst);
        // If the weak count is currently locked, the value of the
        // count was 0 just before taking the lock.
        if cnt == usize::MAX { 0 } else { cnt - 1 }
    }

    #[inline]
    pub fn strong_count(this: &Self) -> usize {
        this.inner().strong.load(SeqCst)
    }

    #[inline]
    fn inner(&self) -> &ArcInner<T> {
        // This unsafety is ok because while this arc is alive we're guaranteed
        // that the inner pointer is valid. Furthermore, we know that the
        // `ArcInner` structure itself is `Sync` because the inner data is
        // `Sync` as well, so we're ok loaning out an immutable pointer to these
        // contents.
        unsafe { self.ptr.as_ref() }
    }

    // Non-inlined part of `drop`.
    #[inline(never)]
    unsafe fn drop_slow(&mut self) {
        // Destroy the data at this time, even though we may not free the box
        // allocation itself (there may still be weak pointers lying around).
        ptr::drop_in_place(&mut self.ptr.as_mut().data);

        if self.inner().weak.fetch_sub(1, Release) == 1 {
            atomic::fence(Acquire);
            System.dealloc(self.ptr.as_ptr().cast(), Layout::for_value(self.ptr.as_ref()))
        }
    }

    #[inline]
    pub fn ptr_eq(this: &Self, other: &Self) -> bool {
        this.ptr.as_ptr() == other.ptr.as_ptr()
    }
}

impl<T: Cacheable> Clone for Arc<T> {
    #[inline]
    fn clone(&self) -> Arc<T> {
        // Using a relaxed ordering is alright here, as knowledge of the
        // original reference prevents other threads from erroneously deleting
        // the object.
        //
        // As explained in the [Boost documentation][1], Increasing the
        // reference counter can always be done with memory_order_relaxed: New
        // references to an object can only be formed from an existing
        // reference, and passing an existing reference from one thread to
        // another must already provide any required synchronization.
        //
        // [1]: (www.boost.org/doc/libs/1_55_0/doc/html/atomic/usage_examples.html)
        let old_size = self.inner().strong.fetch_add(1, Relaxed);

        // However we need to guard against massive refcounts in case someone
        // is `mem::forget`ing Arcs. If we don't do this the count can overflow
        // and users will use-after free. We racily saturate to `isize::MAX` on
        // the assumption that there aren't ~2 billion threads incrementing
        // the reference count at once. This branch will never be taken in
        // any realistic program.
        //
        // We abort because such a program is incredibly degenerate, and we
        // don't care to support it.
        if old_size > MAX_REFCOUNT {
            panic!();
        }

        Self::from_inner(self.ptr)
    }
}

impl<T: Cacheable> Deref for Arc<T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        &self.inner().data
    }
}

impl<T: Clone + Cacheable> Arc<T> {
    #[inline]
    pub fn make_mut(this: &mut Self) -> &mut T {
        // Note that we hold both a strong reference and a weak reference.
        // Thus, releasing our strong reference only will not, by itself, cause
        // the memory to be deallocated.
        //
        // Use Acquire to ensure that we see any writes to `weak` that happen
        // before release writes (i.e., decrements) to `strong`. Since we hold a
        // weak count, there's no chance the ArcInner itself could be
        // deallocated.
        if this.inner().strong.compare_exchange(1, 0, Acquire, Relaxed).is_err() {
            // Another strong pointer exists; clone
            *this = Arc::new_no_cache((**this).clone());
        } else if this.inner().weak.load(Relaxed) != 1 {
            // Relaxed suffices in the above because this is fundamentally an
            // optimization: we are always racing with weak pointers being
            // dropped. Worst case, we end up allocated a new Arc unnecessarily.

            // We removed the last strong ref, but there are additional weak
            // refs remaining. We'll move the contents to a new Arc, and
            // invalidate the other weak refs.

            // Note that it is not possible for the read of `weak` to yield
            // usize::MAX (i.e., locked), since the weak count can only be
            // locked by a thread with a strong reference.

            // Materialize our own implicit weak pointer, so that it can clean
            // up the ArcInner as needed.
            let weak = Weak { ptr: this.ptr };

            // mark the data itself as already deallocated
            unsafe {
                // there is no data race in the implicit write caused by `read`
                // here (due to zeroing) because data is no longer accessed by
                // other threads (due to there being no more strong refs at this
                // point).
                let mut swap = Arc::new_no_cache(ptr::read(&weak.ptr.as_ref().data));
                mem::swap(this, &mut swap);
                mem::forget(swap);
            }
        } else {
            // We were the sole reference of either kind; bump back up the
            // strong ref count.
            this.inner().strong.store(1, Release);
        }

        // As with `get_mut()`, the unsafety is ok because our reference was
        // either unique to begin with, or became one upon cloning the contents.
        unsafe {
            &mut this.ptr.as_mut().data
        }
    }
}

impl<T: Cacheable> Arc<T> {
    #[inline]
    pub fn get_mut(this: &mut Self) -> Option<&mut T> {
        if this.is_unique() {
            // This unsafety is ok because we're guaranteed that the pointer
            // returned is the *only* pointer that will ever be returned to T. Our
            // reference count is guaranteed to be 1 at this point, and we required
            // the Arc itself to be `mut`, so we're returning the only possible
            // reference to the inner data.
            unsafe {
                Some(Arc::get_mut_unchecked(this))
            }
        } else {
            None
        }
    }

    pub unsafe fn get_mut_unchecked(this: &mut Self) -> &mut T {
        &mut this.ptr.as_mut().data
    }

    fn is_unique(&mut self) -> bool {
        // lock the weak pointer count if we appear to be the sole weak pointer
        // holder.
        //
        // The acquire label here ensures a happens-before relationship with any
        // writes to `strong` (in particular in `Weak::upgrade`) prior to decrements
        // of the `weak` count (via `Weak::drop`, which uses release).  If the upgraded
        // weak ref was never dropped, the CAS here will fail so we do not care to synchronize.
        if self.inner().weak.compare_exchange(1, usize::MAX, Acquire, Relaxed).is_ok() {
            // This needs to be an `Acquire` to synchronize with the decrement of the `strong`
            // counter in `drop` -- the only access that happens when any but the last reference
            // is being dropped.
            let unique = self.inner().strong.load(Acquire) == 1;

            // The release write here synchronizes with a read in `downgrade`,
            // effectively preventing the above read of `strong` from happening
            // after the write.
            self.inner().weak.store(1, Release); // release the lock
            unique
        } else {
            false
        }
    }
}

impl<T: Cacheable> Drop for Arc<T> {
    #[inline]
    fn drop(&mut self) {
        // Because `fetch_sub` is already atomic, we do not need to synchronize
        // with other threads unless we are going to delete the object. This
        // same logic applies to the below `fetch_sub` to the `weak` count.
        if self.inner().strong.fetch_sub(1, Release) != 1 {
            return;
        }

        // This fence is needed to prevent reordering of use of the data and
        // deletion of the data.  Because it is marked `Release`, the decreasing
        // of the reference count synchronizes with this `Acquire` fence. This
        // means that use of the data happens before decreasing the reference
        // count, which happens before this fence, which happens before the
        // deletion of the data.
        //
        // As explained in the [Boost documentation][1],
        //
        // > It is important to enforce any possible access to the object in one
        // > thread (through an existing reference) to *happen before* deleting
        // > the object in a different thread. This is achieved by a "release"
        // > operation after dropping a reference (any access to the object
        // > through this reference must obviously happened before), and an
        // > "acquire" operation before deleting the object.
        //
        // In particular, while the contents of an Arc are usually immutable, it's
        // possible to have interior writes to something like a Mutex<T>. Since a
        // Mutex is not acquired when it is deleted, we can't rely on its
        // synchronization logic to make writes in thread A visible to a destructor
        // running in thread B.
        //
        // Also note that the Acquire fence here could probably be replaced with an
        // Acquire load, which could improve performance in highly-contended
        // situations. See [2].
        //
        // [1]: (www.boost.org/doc/libs/1_55_0/doc/html/atomic/usage_examples.html)
        // [2]: (https://github.com/rust-lang/rust/pull/41714)
        atomic::fence(Acquire);

        // See if there is room in the cache for this value, if not
        // proceed with the drop. This must be after the fence so that
        // all other uses of the data are finished when we reinit it.
        if try_put(self) {
            return;
        }
        
        unsafe {
            self.drop_slow();
        }
    }
}

impl<T: Cacheable> Weak<T> {
    pub fn upgrade(&self) -> Option<Arc<T>> {
        // We use a CAS loop to increment the strong count instead of a
        // fetch_add because once the count hits 0 it must never be above 0.
        let inner = self.inner()?;

        // Relaxed load because any write of 0 that we can observe
        // leaves the field in a permanently zero state (so a
        // "stale" read of 0 is fine), and any other value is
        // confirmed via the CAS below.
        let mut n = inner.strong.load(Relaxed);

        loop {
            if n == 0 {
                return None;
            }

            // See comments in `Arc::clone` for why we do this (for `mem::forget`).
            if n > MAX_REFCOUNT {
                panic!();
            }

            // Relaxed is valid for the same reason it is on Arc's Clone impl
            match inner.strong.compare_exchange_weak(n, n + 1, Relaxed, Relaxed) {
                Ok(_) => return Some(Arc::from_inner(self.ptr)), // null checked above
                Err(old) => n = old,
            }
        }
    }

    #[inline]
    fn inner(&self) -> Option<&ArcInner<T>> {
        if is_dangling(self.ptr) {
            None
        } else {
            Some(unsafe { self.ptr.as_ref() })
        }
    }

    #[inline]
    pub fn ptr_eq(&self, other: &Self) -> bool {
        self.ptr.as_ptr() == other.ptr.as_ptr()
    }
}

impl<T: Cacheable> Clone for Weak<T> {
    #[inline]
    fn clone(&self) -> Weak<T> {
        let inner = if let Some(inner) = self.inner() {
            inner
        } else {
            return Weak { ptr: self.ptr };
        };
        // See comments in Arc::clone() for why this is relaxed.  This can use a
        // fetch_add (ignoring the lock) because the weak count is only locked
        // where are *no other* weak pointers in existence. (So we can't be
        // running this code in that case).
        let old_size = inner.weak.fetch_add(1, Relaxed);

        // See comments in Arc::clone() for why we do this (for mem::forget).
        if old_size > MAX_REFCOUNT {
            panic!();
        }

        return Weak { ptr: self.ptr };
    }
}

impl<T: Cacheable> Drop for Weak<T> {
    fn drop(&mut self) {
        // If we find out that we were the last weak pointer, then its time to
        // deallocate the data entirely. See the discussion in Arc::drop() about
        // the memory orderings
        //
        // It's not necessary to check for the locked state here, because the
        // weak count can only be locked if there was precisely one weak ref,
        // meaning that drop could only subsequently run ON that remaining weak
        // ref, which can only happen after the lock is released.
        let inner = if let Some(inner) = self.inner() {
            inner
        } else {
            return
        };

        if inner.weak.fetch_sub(1, Release) == 1 {
            atomic::fence(Acquire);
            unsafe {
                System.dealloc(
                    self.ptr.as_ptr().cast(),
                    Layout::for_value(self.ptr.as_ref())
                )
            }
        }
    }
}

impl<T: PartialEq + Cacheable> PartialEq for Arc<T> {
    #[inline]
    fn eq(&self, other: &Arc<T>) -> bool {
        &**self == &**other
    }

    #[inline]
    fn ne(&self, other: &Arc<T>) -> bool {
        &**self != &**other
    }
}

impl<T: PartialOrd + Cacheable> PartialOrd for Arc<T> {
    fn partial_cmp(&self, other: &Arc<T>) -> Option<Ordering> {
        (**self).partial_cmp(&**other)
    }

    fn lt(&self, other: &Arc<T>) -> bool {
        *(*self) < *(*other)
    }

    fn le(&self, other: &Arc<T>) -> bool {
        *(*self) <= *(*other)
    }

    fn gt(&self, other: &Arc<T>) -> bool {
        *(*self) > *(*other)
    }

    fn ge(&self, other: &Arc<T>) -> bool {
        *(*self) >= *(*other)
    }
}
impl<T: Ord + Cacheable> Ord for Arc<T> {
    fn cmp(&self, other: &Arc<T>) -> Ordering {
        (**self).cmp(&**other)
    }
}
impl<T: Eq + Cacheable> Eq for Arc<T> {}

impl<T: fmt::Display + Cacheable> fmt::Display for Arc<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&**self, f)
    }
}

impl<T: fmt::Debug + Cacheable> fmt::Debug for Arc<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&**self, f)
    }
}

impl<T: Cacheable> fmt::Pointer for Arc<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Pointer::fmt(&(&**self as *const T), f)
    }
}

impl<T: Hash + Cacheable> Hash for Arc<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (**self).hash(state)
    }
}

impl<T: Cacheable> borrow::Borrow<T> for Arc<T> {
    fn borrow(&self) -> &T {
        &**self
    }
}

impl<T: Cacheable> AsRef<T> for Arc<T> {
    fn as_ref(&self) -> &T {
        &**self
    }
}

impl<T: Cacheable> Unpin for Arc<T> { }
