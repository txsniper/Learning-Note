## 介绍
bthread之间的同步是由bthread_mutex完成的(它同样还负责底层的pthread之间的同步)
```
// bthread_mutex_t.butex指向对应的Butex.value
typedef struct {
    unsigned* butex;
    bthread_contention_site_t csite;
} bthread_mutex_t;


struct BAIDU_CACHELINE_ALIGNMENT Butex {
    Butex() {}
    ~Butex() {}

    butil::atomic<int> value;
    ButexWaiterList waiters;
    internal::FastPthreadMutex waiter_lock;
};
```
bthread_mutex_t在上层包装了Butex，对外提供了方便的接口，底层操作全部由Butex完成。

## 创建 (bthread_mutex_create)
bthread_mutex采用了类似于futex的方式，使用一个32bit的数据指针(bthread_mutex_t.butex) 来指代操作的 bthread_mutex
```
int bthread_mutex_init(bthread_mutex_t* __restrict m,
                       const bthread_mutexattr_t* __restrict) __THROW {
    bthread::make_contention_site_invalid(&m->csite);
    m->butex = bthread::butex_create_checked<unsigned>();
    if (!m->butex) {
        return ENOMEM;
    }
    *m->butex = 0;
    return 0;
}

// Check width of user type before casting.
template <typename T> T* butex_create_checked() {
    BAIDU_CASSERT(sizeof(T) == sizeof(int), sizeof_T_must_equal_int);
    return static_cast<T*>(butex_create());
}

void* butex_create() {
    Butex* b = butil::get_object<Butex>();
    if (b) {
        // 返回Butex.value的地址
        return &b->value;
    }
    return NULL;
}
```
可以看到 bthread_mutex_t.butex 指向的实际上是对应的 Butex.value

## 加锁 (lock && trylock && timedlock)
### trylock

先看比较简单的 trylock 函数
```
int bthread_mutex_trylock(bthread_mutex_t* m) __THROW {
    bthread::MutexInternal* split = (bthread::MutexInternal*)m->butex;

    // 利用原子操作的acquire语义来保证内存可见性顺序
    // 如果获得锁，直接返回0，否则返回EBUSY
    if (!split->locked.exchange(1, butil::memory_order_acquire)) {
        return 0;
    }
    return EBUSY;
}

```
它首先将 m->butex 这个指针转化为 bthread::MutexInternal* 类型，实际上该类型占用的字节长度与 m->butex 一致。
```
// Implement bthread_mutex_t related functions
struct MutexInternal {
    butil::static_atomic<unsigned char> locked;         // 加锁状态
    butil::static_atomic<unsigned char> contended;      // 锁冲突
    unsigned short padding;
};

```
与此同时，还定义了两个常量用来代表加锁状态和锁冲突状态，**如果出现锁冲突状态，说明有bthread等待在锁上，在解锁时需要唤醒等待的bthread**
```
const MutexInternal MUTEX_CONTENDED_RAW = {{1},{1},0};
const MutexInternal MUTEX_LOCKED_RAW = {{1},{0},0};
// Define as macros rather than constants which can't be put in read-only
// section and affected by initialization-order fiasco.
#define BTHREAD_MUTEX_CONTENDED (*(const unsigned*)&bthread::MUTEX_CONTENDED_RAW)
#define BTHREAD_MUTEX_LOCKED (*(const unsigned*)&bthread::MUTEX_LOCKED_RAW)

```
### lock 
trylock尝试一次就直接返回，lock函数则会在锁竞争的时候进入等待状态
```
int bthread_mutex_lock(bthread_mutex_t* m) __THROW {
    bthread::MutexInternal* split = (bthread::MutexInternal*)m->butex;

    // step1 : 尝试加锁，成功则返回
    if (!split->locked.exchange(1, butil::memory_order_acquire)) {
        return 0;
    }
    // Don't sample when contention profiler is off.
    // step2 : 如果没有开启锁竞争分析(或者当前需要要采集)，则直接调用  mutex_lock_contended
    if (!bthread::g_cp) {
        return bthread::mutex_lock_contended(m);
    }
    // Ask Collector if this (contended) locking should be sampled.
    const size_t sampling_range = bvar::is_collectable(&bthread::g_cp_sl);
    if (!sampling_range) { // Don't sample
        return bthread::mutex_lock_contended(m);
    }

    // step3 : 开启分析，计算等待锁的时间
    // Start sampling.
    const int64_t start_ns = butil::cpuwide_time_ns();
    // NOTE: Don't modify m->csite outside lock since multiple threads are
    // still contending with each other.
    const int rc = bthread::mutex_lock_contended(m);
    if (!rc) { // Inside lock
        // rc == 0 的情况下说明当前 bthread 获得锁成功
        m->csite.duration_ns = butil::cpuwide_time_ns() - start_ns;
        m->csite.sampling_range = sampling_range;
    } // else rare
    return rc;
}

inline int mutex_lock_contended(bthread_mutex_t* m) {
    butil::atomic<unsigned>* whole = (butil::atomic<unsigned>*)m->butex;
    // 这个while循环条件很巧妙，在锁没有被占用的时候，whole的值为0，因此如果 exchange 返回 0，则说明当前bthread获得锁，while条件不成立,直接返回，如果锁已被占用，则 exchange 返回必不为 0 ，同时将锁设置为 BTHREAD_MUTEX_CONTENDED
    while (whole->exchange(BTHREAD_MUTEX_CONTENDED) & BTHREAD_MUTEX_LOCKED) {

        // 利用 butex_wait 挂起当前 bthread，第三个参数为NULL，代表没有设置超时
        if (bthread::butex_wait(whole, BTHREAD_MUTEX_CONTENDED, NULL) < 0
            && errno != EWOULDBLOCK) {
            return errno;
        }
    }
    return 0;
}

```
当 lock发生锁竞争时，利用 bthread::butex_wait 挂起当前 bthread，具体会在后面进行分析

### timedlock(带超时的锁)
相比起lock会将bthread挂起直到被唤醒，timedlock则带有超时机制，如果到时见后仍没有被唤醒，则退出等待

```
int bthread_mutex_timedlock(bthread_mutex_t* __restrict m,
                            const struct timespec* __restrict abstime) __THROW {
    bthread::MutexInternal* split = (bthread::MutexInternal*)m->butex;

    // step1 : 尝试加锁，如果成功直接返回
    if (!split->locked.exchange(1, butil::memory_order_acquire)) {
        return 0;
    }

    // step2 : 如果不需要做锁竞争分析，直接调用 mutex_timedlock_contended
    // Don't sample when contention profiler is off.
    if (!bthread::g_cp) {
        return bthread::mutex_timedlock_contended(m, abstime);
    }
    // Ask Collector if this (contended) locking should be sampled.
    const size_t sampling_range = bvar::is_collectable(&bthread::g_cp_sl);
    if (!sampling_range) { // Don't sample
        return bthread::mutex_timedlock_contended(m, abstime);
    }

    // step3 : 获得等待锁的时间
    // Start sampling.
    const int64_t start_ns = butil::cpuwide_time_ns();
    // NOTE: Don't modify m->csite outside lock since multiple threads are
    // still contending with each other.
    const int rc = bthread::mutex_timedlock_contended(m, abstime);
    if (!rc) { // Inside lock
        m->csite.duration_ns = butil::cpuwide_time_ns() - start_ns;
        m->csite.sampling_range = sampling_range;
    } else if (rc == ETIMEDOUT) {
        // Failed to lock due to ETIMEDOUT, submit the elapse directly.
        // 注意 ： 超时返回时当前的 bthread 并没有获得锁
        const int64_t end_ns = butil::cpuwide_time_ns();
        const bthread_contention_site_t csite = {end_ns - start_ns, sampling_range};
        bthread::submit_contention(csite, end_ns);
    }
    return rc;
}
```

## 解锁 (unlock)
解锁，根据锁的状态确定是否需要唤醒等待的bthread
```
int bthread_mutex_unlock(bthread_mutex_t* m) __THROW {
    butil::atomic<unsigned>* whole = (butil::atomic<unsigned>*)m->butex;
    bthread_contention_site_t saved_csite = {0, 0};
    if (bthread::is_contention_site_valid(m->csite)) {
        saved_csite = m->csite;
        bthread::make_contention_site_invalid(&m->csite);
    }
    // step1 : 解除加锁状态, 获取解锁前的状态
    const unsigned prev = whole->exchange(0, butil::memory_order_release);
    // CAUTION: the mutex may be destroyed, check comments before butex_create

    // 如果之前的状态只是加锁，说明没有其他bthread等待，直接返回
    if (prev == BTHREAD_MUTEX_LOCKED) {
        return 0;
    }
    // Wakeup one waiter
    // step2 : 如果之前的状态是 BTHREAD_MUTEX_CONTENDED , 则唤醒一个等待的bthread
    if (!bthread::is_contention_site_valid(saved_csite)) {
        bthread::butex_wake(whole);
        return 0;
    }
    const int64_t unlock_start_ns = butil::cpuwide_time_ns();
    bthread::butex_wake(whole);
    const int64_t unlock_end_ns = butil::cpuwide_time_ns();
    saved_csite.duration_ns += unlock_end_ns - unlock_start_ns;
    bthread::submit_contention(saved_csite, unlock_end_ns);
    return 0;
}
```

