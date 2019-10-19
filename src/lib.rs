use std::{
    borrow, fmt, mem, isize, usize,
    sync::atomic::{self, Ordering::{Acquire, Relaxed, Release, SeqCst}},
    cmp::Ordering,
    ops::Deref,
    ptr::{self, NonNull},
    marker::{Unpin, PhantomData},
    hash::{Hash, Hasher},    
    alloc::{System, GlobalAlloc, Layout},
    boxed::Box,
    collections::HashMap,
    any::{TypeId, Any},
    cell::RefCell,
};

static LIMIT: usize = 10_000;

//#[cfg(test)]
//mod tests;
struct PoolByType {
    clear: usize,
    vals: Vec<usize>,
}

thread_local! {
    static POOL: RefCell<HashMap<TypeId, PoolByType>> = RefCell::new(HashMap::new());
}

fn raeify<T: Any>(ptr: usize) -> Arc<T> {
    let t = unsafe {
        mem::transmute::<usize, NonNull<ArcInner<T>>>(ptr)
    };
    let t = Arc::from_inner(t);
    t.inner().strong.fetch_add(1, Relaxed);
    t
}

pub fn clear_cache<T: Any>() {
    let tid = TypeId::of::<T>();
    POOL.with(|inner| {
        let mut inner = inner.borrow_mut();
        if let Some(pool) = inner.get_mut(&tid) {
            pool.clear = pool.vals.len();
            for v in pool.vals.drain(0..) {
                raeify::<T>(v);
            }
        }
    })
}

fn take<T: Any>() -> Option<Arc<T>> {
    let tid = TypeId::of::<T>();
    POOL.with(|inner| {
        let mut inner = inner.borrow_mut();
        inner.get_mut(&tid).and_then(|p| p.vals.pop().map(raeify::<T>))
    })
}

fn put<T: Any>(v: &Arc<T>) -> bool {
    let tid = TypeId::of::<T>();
    POOL.with(|inner| {
        let mut inner = inner.borrow_mut();
        let pool = inner.entry(tid).or_insert_with(|| {
            PoolByType { clear: 0, vals: Vec::new() }
        });
        if pool.vals.len() < LIMIT && pool.clear == 0 {
            let t = unsafe {
                mem::transmute::<NonNull<ArcInner<T>>, usize>(v.ptr)
            };
            pool.vals.push(t);
            true
        } else {
            if pool.clear > 0 {
                pool.clear -= 1
            }
            false
        }
    })
}

/// A soft limit on the amount of references that may be made to an `Arc`.
///
/// Going above this limit will abort your program (although not
/// necessarily) at _exactly_ `MAX_REFCOUNT + 1` references.
const MAX_REFCOUNT: usize = (isize::MAX) as usize;

fn is_dangling<T: ?Sized>(ptr: NonNull<T>) -> bool {
    let address = ptr.as_ptr() as *mut () as usize;
    address == usize::MAX
}

/// A thread-safe reference-counting pointer. 'Arc' stands for 'Atomically
/// Reference Counted'.
///
/// The type `Arc<T>` provides shared ownership of a value of type `T`,
/// allocated in the heap. Invoking [`clone`][clone] on `Arc` produces
/// a new `Arc` instance, which points to the same value on the heap as the
/// source `Arc`, while increasing a reference count. When the last `Arc`
/// pointer to a given value is destroyed, the pointed-to value is also
/// destroyed.
///
/// Shared references in Rust disallow mutation by default, and `Arc` is no
/// exception: you cannot generally obtain a mutable reference to something
/// inside an `Arc`. If you need to mutate through an `Arc`, use
/// [`Mutex`][mutex], [`RwLock`][rwlock], or one of the [`Atomic`][atomic]
/// types.
///
/// ## Thread Safety
///
/// Unlike [`Rc<T>`], `Arc<T>` uses atomic operations for its reference
/// counting. This means that it is thread-safe. The disadvantage is that
/// atomic operations are more expensive than ordinary memory accesses. If you
/// are not sharing reference-counted values between threads, consider using
/// [`Rc<T>`] for lower overhead. [`Rc<T>`] is a safe default, because the
/// compiler will catch any attempt to send an [`Rc<T>`] between threads.
/// However, a library might choose `Arc<T>` in order to give library consumers
/// more flexibility.
///
/// `Arc<T>` will implement [`Send`] and [`Sync`] as long as the `T` implements
/// [`Send`] and [`Sync`]. Why can't you put a non-thread-safe type `T` in an
/// `Arc<T>` to make it thread-safe? This may be a bit counter-intuitive at
/// first: after all, isn't the point of `Arc<T>` thread safety? The key is
/// this: `Arc<T>` makes it thread safe to have multiple ownership of the same
/// data, but it  doesn't add thread safety to its data. Consider
/// `Arc<`[`RefCell<T>`]`>`. [`RefCell<T>`] isn't [`Sync`], and if `Arc<T>` was always
/// [`Send`], `Arc<`[`RefCell<T>`]`>` would be as well. But then we'd have a problem:
/// [`RefCell<T>`] is not thread safe; it keeps track of the borrowing count using
/// non-atomic operations.
///
/// In the end, this means that you may need to pair `Arc<T>` with some sort of
/// [`std::sync`] type, usually [`Mutex<T>`][mutex].
///
/// ## Breaking cycles with `Weak`
///
/// The [`downgrade`][downgrade] method can be used to create a non-owning
/// [`Weak`][weak] pointer. A [`Weak`][weak] pointer can be [`upgrade`][upgrade]d
/// to an `Arc`, but this will return [`None`] if the value has already been
/// dropped.
///
/// A cycle between `Arc` pointers will never be deallocated. For this reason,
/// [`Weak`][weak] is used to break cycles. For example, a tree could have
/// strong `Arc` pointers from parent nodes to children, and [`Weak`][weak]
/// pointers from children back to their parents.
///
/// # Cloning references
///
/// Creating a new reference from an existing reference counted pointer is done using the
/// `Clone` trait implemented for [`Arc<T>`][arc] and [`Weak<T>`][weak].
///
/// ```
/// use std::sync::Arc;
/// let foo = Arc::new(vec![1.0, 2.0, 3.0]);
/// // The two syntaxes below are equivalent.
/// let a = foo.clone();
/// let b = Arc::clone(&foo);
/// // a, b, and foo are all Arcs that point to the same memory location
/// ```
///
/// ## `Deref` behavior
///
/// `Arc<T>` automatically dereferences to `T` (via the [`Deref`][deref] trait),
/// so you can call `T`'s methods on a value of type `Arc<T>`. To avoid name
/// clashes with `T`'s methods, the methods of `Arc<T>` itself are associated
/// functions, called using function-like syntax:
///
/// ```
/// use std::sync::Arc;
/// let my_arc = Arc::new(());
///
/// Arc::downgrade(&my_arc);
/// ```
///
/// [`Weak<T>`][weak] does not auto-dereference to `T`, because the value may have
/// already been destroyed.
///
/// [arc]: struct.Arc.html
/// [weak]: struct.Weak.html
/// [`Rc<T>`]: ../../std/rc/struct.Rc.html
/// [clone]: ../../std/clone/trait.Clone.html#tymethod.clone
/// [mutex]: ../../std/sync/struct.Mutex.html
/// [rwlock]: ../../std/sync/struct.RwLock.html
/// [atomic]: ../../std/sync/atomic/index.html
/// [`Send`]: ../../std/marker/trait.Send.html
/// [`Sync`]: ../../std/marker/trait.Sync.html
/// [deref]: ../../std/ops/trait.Deref.html
/// [downgrade]: struct.Arc.html#method.downgrade
/// [upgrade]: struct.Weak.html#method.upgrade
/// [`None`]: ../../std/option/enum.Option.html#variant.None
/// [`RefCell<T>`]: ../../std/cell/struct.RefCell.html
/// [`std::sync`]: ../../std/sync/index.html
/// [`Arc::clone(&from)`]: #method.clone
///
/// # Examples
///
/// Sharing some immutable data between threads:
///
// Note that we **do not** run these tests here. The windows builders get super
// unhappy if a thread outlives the main thread and then exits at the same time
// (something deadlocks) so we just avoid this entirely by not running these
// tests.
/// ```no_run
/// use std::sync::Arc;
/// use std::thread;
///
/// let five = Arc::new(5);
///
/// for _ in 0..10 {
///     let five = Arc::clone(&five);
///
///     thread::spawn(move || {
///         println!("{:?}", five);
///     });
/// }
/// ```
///
/// Sharing a mutable [`AtomicUsize`]:
///
/// [`AtomicUsize`]: ../../std/sync/atomic/struct.AtomicUsize.html
///
/// ```no_run
/// use std::sync::Arc;
/// use std::sync::atomic::{AtomicUsize, Ordering};
/// use std::thread;
///
/// let val = Arc::new(AtomicUsize::new(5));
///
/// for _ in 0..10 {
///     let val = Arc::clone(&val);
///
///     thread::spawn(move || {
///         let v = val.fetch_add(1, Ordering::SeqCst);
///         println!("{:?}", v);
///     });
/// }
/// ```
///
/// See the [`rc` documentation][rc_examples] for more examples of reference
/// counting in general.
///
/// [rc_examples]: ../../std/rc/index.html#examples
pub struct Arc<T: Any> {
    ptr: NonNull<ArcInner<T>>,
    phantom: PhantomData<T>,
}

unsafe impl<T: Sync + Send + Any> Send for Arc<T> {}
unsafe impl<T: Sync + Send + Any> Sync for Arc<T> {}

impl<T: Any> Arc<T> {
    fn from_inner(ptr: NonNull<ArcInner<T>>) -> Self {
        Self {
            ptr,
            phantom: PhantomData,
        }
    }
}

/// `Weak` is a version of [`Arc`] that holds a non-owning reference to the
/// managed value. The value is accessed by calling [`upgrade`] on the `Weak`
/// pointer, which returns an [`Option`]`<`[`Arc`]`<T>>`.
///
/// Since a `Weak` reference does not count towards ownership, it will not
/// prevent the inner value from being dropped, and `Weak` itself makes no
/// guarantees about the value still being present and may return [`None`]
/// when [`upgrade`]d.
///
/// A `Weak` pointer is useful for keeping a temporary reference to the value
/// within [`Arc`] without extending its lifetime. It is also used to prevent
/// circular references between [`Arc`] pointers, since mutual owning references
/// would never allow either [`Arc`] to be dropped. For example, a tree could
/// have strong [`Arc`] pointers from parent nodes to children, and `Weak`
/// pointers from children back to their parents.
///
/// The typical way to obtain a `Weak` pointer is to call [`Arc::downgrade`].
///
/// [`Arc`]: struct.Arc.html
/// [`Arc::downgrade`]: struct.Arc.html#method.downgrade
/// [`upgrade`]: struct.Weak.html#method.upgrade
/// [`Option`]: ../../std/option/enum.Option.html
/// [`None`]: ../../std/option/enum.Option.html#variant.None
pub struct Weak<T: Any> {
    // This is a `NonNull` to allow optimizing the size of this type in enums,
    // but it is not necessarily a valid pointer.
    // `Weak::new` sets this to `usize::MAX` so that it doesn’t need
    // to allocate space on the heap.  That's not a value a real pointer
    // will ever have because RcBox has alignment at least 2.
    ptr: NonNull<ArcInner<T>>,
}

unsafe impl<T: Sync + Send + Any> Send for Weak<T> {}
unsafe impl<T: Sync + Send + Any> Sync for Weak<T> {}

impl<T: fmt::Debug + Any> fmt::Debug for Weak<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(Weak)")
    }
}

struct ArcInner<T: Any> {
    strong: atomic::AtomicUsize,
    // the value usize::MAX acts as a sentinel for temporarily "locking" the
    // ability to upgrade weak pointers or downgrade strong ones; this is used
    // to avoid races in `make_mut` and `get_mut`.
    weak: atomic::AtomicUsize,
    data: T,
}

unsafe impl<T: Sync + Send + Any> Send for ArcInner<T> {}
unsafe impl<T: Sync + Send + Any> Sync for ArcInner<T> {}

impl<T: Any> Arc<T> {
    /// Constructs a new `Arc<T>`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// let five = Arc::new(5);
    /// ```
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

    pub fn new<F: FnOnce() -> T>(f: F) -> Arc<T> {
        take::<T>().unwrap_or_else(|| Arc::new_no_cache(f()))
    }

    /// Returns the contained value, if the `Arc` has exactly one strong reference.
    ///
    /// Otherwise, an [`Err`][result] is returned with the same `Arc` that was
    /// passed in.
    ///
    /// This will succeed even if there are outstanding weak references.
    ///
    /// [result]: ../../std/result/enum.Result.html
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// let x = Arc::new(3);
    /// assert_eq!(Arc::try_unwrap(x), Ok(3));
    ///
    /// let x = Arc::new(4);
    /// let _y = Arc::clone(&x);
    /// assert_eq!(*Arc::try_unwrap(x).unwrap_err(), 4);
    /// ```
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

impl<T: Any> Arc<T> {
    /// Consumes the `Arc`, returning the wrapped pointer.
    ///
    /// To avoid a memory leak the pointer must be converted back to an `Arc` using
    /// [`Arc::from_raw`][from_raw].
    ///
    /// [from_raw]: struct.Arc.html#method.from_raw
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// let x = Arc::new("hello".to_owned());
    /// let x_ptr = Arc::into_raw(x);
    /// assert_eq!(unsafe { &*x_ptr }, "hello");
    /// ```
    pub fn into_raw(this: Self) -> *const T {
        let ptr: *const T = &*this;
        mem::forget(this);
        ptr
    }

    /// Creates a new [`Weak`][weak] pointer to this value.
    ///
    /// [weak]: struct.Weak.html
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// let five = Arc::new(5);
    ///
    /// let weak_five = Arc::downgrade(&five);
    /// ```
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

    /// Gets the number of [`Weak`][weak] pointers to this value.
    ///
    /// [weak]: struct.Weak.html
    ///
    /// # Safety
    ///
    /// This method by itself is safe, but using it correctly requires extra care.
    /// Another thread can change the weak count at any time,
    /// including potentially between calling this method and acting on the result.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// let five = Arc::new(5);
    /// let _weak_five = Arc::downgrade(&five);
    ///
    /// // This assertion is deterministic because we haven't shared
    /// // the `Arc` or `Weak` between threads.
    /// assert_eq!(1, Arc::weak_count(&five));
    /// ```
    #[inline]
    pub fn weak_count(this: &Self) -> usize {
        let cnt = this.inner().weak.load(SeqCst);
        // If the weak count is currently locked, the value of the
        // count was 0 just before taking the lock.
        if cnt == usize::MAX { 0 } else { cnt - 1 }
    }

    /// Gets the number of strong (`Arc`) pointers to this value.
    ///
    /// # Safety
    ///
    /// This method by itself is safe, but using it correctly requires extra care.
    /// Another thread can change the strong count at any time,
    /// including potentially between calling this method and acting on the result.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// let five = Arc::new(5);
    /// let _also_five = Arc::clone(&five);
    ///
    /// // This assertion is deterministic because we haven't shared
    /// // the `Arc` between threads.
    /// assert_eq!(2, Arc::strong_count(&five));
    /// ```
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
    /// Returns `true` if the two `Arc`s point to the same value (not
    /// just values that compare as equal).
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// let five = Arc::new(5);
    /// let same_five = Arc::clone(&five);
    /// let other_five = Arc::new(5);
    ///
    /// assert!(Arc::ptr_eq(&five, &same_five));
    /// assert!(!Arc::ptr_eq(&five, &other_five));
    /// ```
    pub fn ptr_eq(this: &Self, other: &Self) -> bool {
        this.ptr.as_ptr() == other.ptr.as_ptr()
    }
}

impl<T: Any> Clone for Arc<T> {
    /// Makes a clone of the `Arc` pointer.
    ///
    /// This creates another pointer to the same inner value, increasing the
    /// strong reference count.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// let five = Arc::new(5);
    ///
    /// let _ = Arc::clone(&five);
    /// ```
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

impl<T: Any> Deref for Arc<T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        &self.inner().data
    }
}

impl<T: Clone + Any> Arc<T> {
    /// Makes a mutable reference into the given `Arc`.
    ///
    /// If there are other `Arc` or [`Weak`][weak] pointers to the same value,
    /// then `make_mut` will invoke [`clone`][clone] on the inner value to
    /// ensure unique ownership. This is also referred to as clone-on-write.
    ///
    /// See also [`get_mut`][get_mut], which will fail rather than cloning.
    ///
    /// [weak]: struct.Weak.html
    /// [clone]: ../../std/clone/trait.Clone.html#tymethod.clone
    /// [get_mut]: struct.Arc.html#method.get_mut
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// let mut data = Arc::new(5);
    ///
    /// *Arc::make_mut(&mut data) += 1;         // Won't clone anything
    /// let mut other_data = Arc::clone(&data); // Won't clone inner data
    /// *Arc::make_mut(&mut data) += 1;         // Clones inner data
    /// *Arc::make_mut(&mut data) += 1;         // Won't clone anything
    /// *Arc::make_mut(&mut other_data) *= 2;   // Won't clone anything
    ///
    /// // Now `data` and `other_data` point to different values.
    /// assert_eq!(*data, 8);
    /// assert_eq!(*other_data, 12);
    /// ```
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

impl<T: Any> Arc<T> {
    /// Returns a mutable reference to the inner value, if there are
    /// no other `Arc` or [`Weak`][weak] pointers to the same value.
    ///
    /// Returns [`None`][option] otherwise, because it is not safe to
    /// mutate a shared value.
    ///
    /// See also [`make_mut`][make_mut], which will [`clone`][clone]
    /// the inner value when it's shared.
    ///
    /// [weak]: struct.Weak.html
    /// [option]: ../../std/option/enum.Option.html
    /// [make_mut]: struct.Arc.html#method.make_mut
    /// [clone]: ../../std/clone/trait.Clone.html#tymethod.clone
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// let mut x = Arc::new(3);
    /// *Arc::get_mut(&mut x).unwrap() = 4;
    /// assert_eq!(*x, 4);
    ///
    /// let _y = Arc::clone(&x);
    /// assert!(Arc::get_mut(&mut x).is_none());
    /// ```
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

    /// Determine whether this is the unique reference (including weak refs) to
    /// the underlying data.
    ///
    /// Note that this requires locking the weak ref count.
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

impl<T: Any> Drop for Arc<T> {
    /// Drops the `Arc`.
    ///
    /// This will decrement the strong reference count. If the strong reference
    /// count reaches zero then the only other references (if any) are
    /// [`Weak`], so we `drop` the inner value.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// struct Foo;
    ///
    /// impl Drop for Foo {
    ///     fn drop(&mut self) {
    ///         println!("dropped!");
    ///     }
    /// }
    ///
    /// let foo  = Arc::new(Foo);
    /// let foo2 = Arc::clone(&foo);
    ///
    /// drop(foo);    // Doesn't print anything
    /// drop(foo2);   // Prints "dropped!"
    /// ```
    ///
    /// [`Weak`]: ../../std/sync/struct.Weak.html
    #[inline]
    fn drop(&mut self) {
        // Because `fetch_sub` is already atomic, we do not need to synchronize
        // with other threads unless we are going to delete the object. This
        // same logic applies to the below `fetch_sub` to the `weak` count.
        if self.inner().strong.fetch_sub(1, Release) != 1 {
            return;
        }

        // See if there is room in the cache for this value, if not
        // proceed with the drop.
        if put(self) {
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
        
        unsafe {
            self.drop_slow();
        }
    }
}

impl<T: Any> Weak<T> {
    /// Attempts to upgrade the `Weak` pointer to an [`Arc`], extending
    /// the lifetime of the value if successful.
    ///
    /// Returns [`None`] if the value has since been dropped.
    ///
    /// [`Arc`]: struct.Arc.html
    /// [`None`]: ../../std/option/enum.Option.html#variant.None
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// let five = Arc::new(5);
    ///
    /// let weak_five = Arc::downgrade(&five);
    ///
    /// let strong_five: Option<Arc<_>> = weak_five.upgrade();
    /// assert!(strong_five.is_some());
    ///
    /// // Destroy all strong pointers.
    /// drop(strong_five);
    /// drop(five);
    ///
    /// assert!(weak_five.upgrade().is_none());
    /// ```
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

    /// Returns `None` when the pointer is dangling and there is no allocated `ArcInner`,
    /// (i.e., when this `Weak` was created by `Weak::new`).
    #[inline]
    fn inner(&self) -> Option<&ArcInner<T>> {
        if is_dangling(self.ptr) {
            None
        } else {
            Some(unsafe { self.ptr.as_ref() })
        }
    }

    /// Returns `true` if the two `Weak`s point to the same value (not just
    /// values that compare as equal), or if both don't point to any value
    /// (because they were created with `Weak::new()`).
    ///
    /// # Notes
    ///
    /// Since this compares pointers it means that `Weak::new()` will equal each
    /// other, even though they don't point to any value.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// let first_rc = Arc::new(5);
    /// let first = Arc::downgrade(&first_rc);
    /// let second = Arc::downgrade(&first_rc);
    ///
    /// assert!(first.ptr_eq(&second));
    ///
    /// let third_rc = Arc::new(5);
    /// let third = Arc::downgrade(&third_rc);
    ///
    /// assert!(!first.ptr_eq(&third));
    /// ```
    ///
    /// Comparing `Weak::new`.
    ///
    /// ```
    /// use std::sync::{Arc, Weak};
    ///
    /// let first = Weak::new();
    /// let second = Weak::new();
    /// assert!(first.ptr_eq(&second));
    ///
    /// let third_rc = Arc::new(());
    /// let third = Arc::downgrade(&third_rc);
    /// assert!(!first.ptr_eq(&third));
    /// ```
    #[inline]
    pub fn ptr_eq(&self, other: &Self) -> bool {
        self.ptr.as_ptr() == other.ptr.as_ptr()
    }
}

impl<T: Any> Clone for Weak<T> {
    /// Makes a clone of the `Weak` pointer that points to the same value.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::{Arc, Weak};
    ///
    /// let weak_five = Arc::downgrade(&Arc::new(5));
    ///
    /// let _ = Weak::clone(&weak_five);
    /// ```
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

impl<T: Any> Drop for Weak<T> {
    /// Drops the `Weak` pointer.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::{Arc, Weak};
    ///
    /// struct Foo;
    ///
    /// impl Drop for Foo {
    ///     fn drop(&mut self) {
    ///         println!("dropped!");
    ///     }
    /// }
    ///
    /// let foo = Arc::new(Foo);
    /// let weak_foo = Arc::downgrade(&foo);
    /// let other_weak_foo = Weak::clone(&weak_foo);
    ///
    /// drop(weak_foo);   // Doesn't print anything
    /// drop(foo);        // Prints "dropped!"
    ///
    /// assert!(other_weak_foo.upgrade().is_none());
    /// ```
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

impl<T: PartialEq + Any> PartialEq for Arc<T> {
    /// Equality for two `Arc`s.
    ///
    /// Two `Arc`s are equal if their inner values are equal.
    ///
    /// If `T` also implements `Eq`, two `Arc`s that point to the same value are
    /// always equal.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// let five = Arc::new(5);
    ///
    /// assert!(five == Arc::new(5));
    /// ```
    #[inline]
    fn eq(&self, other: &Arc<T>) -> bool {
        &**self == &**other
    }

    /// Inequality for two `Arc`s.
    ///
    /// Two `Arc`s are unequal if their inner values are unequal.
    ///
    /// If `T` also implements `Eq`, two `Arc`s that point to the same value are
    /// never unequal.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// let five = Arc::new(5);
    ///
    /// assert!(five != Arc::new(6));
    /// ```
    #[inline]
    fn ne(&self, other: &Arc<T>) -> bool {
        &**self != &**other
    }
}

impl<T: PartialOrd + Any> PartialOrd for Arc<T> {
    /// Partial comparison for two `Arc`s.
    ///
    /// The two are compared by calling `partial_cmp()` on their inner values.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    /// use std::cmp::Ordering;
    ///
    /// let five = Arc::new(5);
    ///
    /// assert_eq!(Some(Ordering::Less), five.partial_cmp(&Arc::new(6)));
    /// ```
    fn partial_cmp(&self, other: &Arc<T>) -> Option<Ordering> {
        (**self).partial_cmp(&**other)
    }

    /// Less-than comparison for two `Arc`s.
    ///
    /// The two are compared by calling `<` on their inner values.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// let five = Arc::new(5);
    ///
    /// assert!(five < Arc::new(6));
    /// ```
    fn lt(&self, other: &Arc<T>) -> bool {
        *(*self) < *(*other)
    }

    /// 'Less than or equal to' comparison for two `Arc`s.
    ///
    /// The two are compared by calling `<=` on their inner values.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// let five = Arc::new(5);
    ///
    /// assert!(five <= Arc::new(5));
    /// ```
    fn le(&self, other: &Arc<T>) -> bool {
        *(*self) <= *(*other)
    }

    /// Greater-than comparison for two `Arc`s.
    ///
    /// The two are compared by calling `>` on their inner values.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// let five = Arc::new(5);
    ///
    /// assert!(five > Arc::new(4));
    /// ```
    fn gt(&self, other: &Arc<T>) -> bool {
        *(*self) > *(*other)
    }

    /// 'Greater than or equal to' comparison for two `Arc`s.
    ///
    /// The two are compared by calling `>=` on their inner values.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// let five = Arc::new(5);
    ///
    /// assert!(five >= Arc::new(5));
    /// ```
    fn ge(&self, other: &Arc<T>) -> bool {
        *(*self) >= *(*other)
    }
}
impl<T: Ord + Any> Ord for Arc<T> {
    /// Comparison for two `Arc`s.
    ///
    /// The two are compared by calling `cmp()` on their inner values.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    /// use std::cmp::Ordering;
    ///
    /// let five = Arc::new(5);
    ///
    /// assert_eq!(Ordering::Less, five.cmp(&Arc::new(6)));
    /// ```
    fn cmp(&self, other: &Arc<T>) -> Ordering {
        (**self).cmp(&**other)
    }
}
impl<T: Eq + Any> Eq for Arc<T> {}

impl<T: fmt::Display + Any> fmt::Display for Arc<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&**self, f)
    }
}

impl<T: fmt::Debug + Any> fmt::Debug for Arc<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&**self, f)
    }
}

impl<T: Any> fmt::Pointer for Arc<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Pointer::fmt(&(&**self as *const T), f)
    }
}

impl<T: Hash + Any> Hash for Arc<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (**self).hash(state)
    }
}

impl<T: Any> borrow::Borrow<T> for Arc<T> {
    fn borrow(&self) -> &T {
        &**self
    }
}

impl<T: Any> AsRef<T> for Arc<T> {
    fn as_ref(&self) -> &T {
        &**self
    }
}

impl<T: Any> Unpin for Arc<T> { }
