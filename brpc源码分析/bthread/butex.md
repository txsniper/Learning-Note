## 介绍
bthread_mutex_t的底层使用butex来完成锁的功能，对于一把锁来说，主要完成两个功能：  
1. 利用状态同步不同执行流之间的操作
2. 当多个执行流对同一把锁加锁时，没有获得锁的执行流挂起等待，获得锁的执行流会在释放锁之后唤醒挂起等待的执行流

## 对外接口
butex的对外接口声明在butex.h中，从中可以看出butex提供的大体功能：
1. butex_create (butex_create_checked) : 创建 butex
2. butex_destroy : 销毁 butex
3. butex_wake (butex_wake_all, butex_wake_except) : 唤醒
4. butex_requeue : 唤醒一个，转移剩下的，详细见注释
5. butex_wait (butex_wait_uninterruptible) : 等待
```
// Create a butex which is a futex-like 32-bit primitive for synchronizing
// bthreads/pthreads.
// Returns a pointer to 32-bit data, NULL on failure.
// NOTE: all butexes are private(not inter-process).
void* butex_create();

// Check width of user type before casting.
template <typename T> T* butex_create_checked() {
    BAIDU_CASSERT(sizeof(T) == sizeof(int), sizeof_T_must_equal_int);
    return static_cast<T*>(butex_create());
}

// Destroy the butex.
void butex_destroy(void* butex);

// Wake up at most 1 thread waiting on |butex|.
// Returns # of threads woken up.
int butex_wake(void* butex);

// Wake up all threads waiting on |butex|.
// Returns # of threads woken up.
int butex_wake_all(void* butex);

// Wake up all threads waiting on |butex| except a bthread whose identifier
// is |excluded_bthread|. This function does not yield.
// Returns # of threads woken up.
int butex_wake_except(void* butex, bthread_t excluded_bthread);

// Wake up at most 1 thread waiting on |butex1|, let all other threads wait
// on |butex2| instead.
// Returns # of threads woken up.
int butex_requeue(void* butex1, void* butex2);

// Atomically wait on |butex| if *butex equals |expected_value|, until the
// butex is woken up by butex_wake*, or CLOCK_REALTIME reached |abstime| if
// abstime is not NULL.
// About |abstime|:
//   Different from FUTEX_WAIT, butex_wait uses absolute time.
int butex_wait(void* butex, int expected_value, const timespec* abstime);

// Same with butex_wait except that this function cannot be woken up by
// bthread_stop(), although this function still returns -1(ESTOP) after
// wake-up.
int butex_wait_uninterruptible(void* butex, int expected_value,
                               const timespec* abstime);

```

## 数据结构

```
struct BAIDU_CACHELINE_ALIGNMENT Butex {
    Butex() {}
    ~Butex() {}

    butil::atomic<int> value;               // 存储锁的状态，详细见bthread_mutex_t
    ButexWaiterList waiters;                // 等待在Butex上的bthread或pthread
    internal::FastPthreadMutex waiter_lock; // 保护waiters的锁
};


```
**注意 waiter_lock 是用来同步底层的 pthread的，防止运行在不同pthread的bthread同时更改 waiters，对于运行在相同pthread上的 bthread，不会同时更新 waiters**

ButexWaiter指代一个等待Butex的对象：bthread或者pthread，具体的数据结构为ButexBthreadWaiter和ButexBthreadWaiter
```
struct ButexWaiter : public butil::LinkNode<ButexWaiter> {
    // tids of pthreads are 0
    bthread_t tid;

    // Erasing node from middle of LinkedList is thread-unsafe, we need
    // to hold its container's lock.
    butil::atomic<Butex*> container;
};

// non_pthread_task allocates this structure on stack and queue it in
// Butex::waiters.
struct ButexBthreadWaiter : public ButexWaiter {
    TaskMeta* task_meta;
    TimerThread::TaskId sleep_id;
    WaiterState waiter_state;
    int expected_value;
    Butex* initial_butex;
    TaskControl* control;
};

// pthread_task or main_task allocates this structure on stack and queue it
// in Butex::waiters.
struct ButexPthreadWaiter : public ButexWaiter {
    butil::atomic<int> sig;
};

typedef butil::LinkedList<ButexWaiter> ButexWaiterList;

```

## 创建和销毁
Butex使用预先的对象池分配，在销毁后会返还给对象池，这样做的目的有两个：  
1. 加快分配和回收的速度，不从操作系统分配。
2. **防止创建锁的线程在销毁之后，其他线程访问锁导致Crash，这个问题会在下面详细解释**
```
void* butex_create() {
    Butex* b = butil::get_object<Butex>();
    if (b) {
        return &b->value;
    }
    return NULL;
}

void butex_destroy(void* butex) {
    if (!butex) {
        return;
    }
    Butex* b = static_cast<Butex*>(
        container_of(static_cast<butil::atomic<int>*>(butex), Butex, value));
    butil::return_object(b);
}
```
## 等待操作
当前的bthread通过调用butex_wait来等待条件的触发
```
int butex_wait(void* arg, int expected_value, const timespec* abstime) {
    Butex* b = container_of(static_cast<butil::atomic<int>*>(arg), Butex, value);
    if (b->value.load(butil::memory_order_relaxed) != expected_value) {
        errno = EWOULDBLOCK;
        // Sometimes we may take actions immediately after unmatched butex,
        // this fence makes sure that we see changes before changing butex.
        // 这里的fence保证当前的bthread在看到b->value的新值时能够看到
        // 其他的变化
        butil::atomic_thread_fence(butil::memory_order_acquire);
        return -1;
    }
    TaskGroup* g = tls_task_group;
    // 如果当前的bthread是main bthread，则调用wait_from_pthread
    if (NULL == g || g->is_current_pthread_task()) {
        return butex_wait_from_pthread(g, b, expected_value, abstime);
    }
    ButexBthreadWaiter bbw;
    // tid is 0 iff the thread is non-bthread
    bbw.tid = g->current_tid();
    bbw.container.store(NULL, butil::memory_order_relaxed);
    bbw.task_meta = g->current_task();
    bbw.sleep_id = 0;
    bbw.waiter_state = WAITER_STATE_READY;
    bbw.expected_value = expected_value;
    bbw.initial_butex = b;
    bbw.control = g->control();

    if (abstime != NULL) {
        // Schedule timer before queueing. If the timer is triggered before
        // queueing, cancel queueing. This is a kind of optimistic locking.
        if (butil::timespec_to_microseconds(*abstime) <
            (butil::gettimeofday_us() + MIN_SLEEP_US)) {
            // Already timed out.
            errno = ETIMEDOUT;
            return -1;
        }

        // 如果有超时设置，则加入到TimerThread中，TimerThread在
        // 超时的时候会调用 erase_from_butex_and_wakeup唤醒等待的
        // bthread
        bbw.sleep_id = get_global_timer_thread()->schedule(
            erase_from_butex_and_wakeup, &bbw, *abstime);
        if (!bbw.sleep_id) {  // TimerThread stopped.
            errno = ESTOP;
            return -1;
        }
    }
#ifdef SHOW_BTHREAD_BUTEX_WAITER_COUNT_IN_VARS
    bvar::Adder<int64_t>& num_waiters = butex_waiter_count();
    num_waiters << 1;
#endif

    // release fence matches with acquire fence in interrupt_and_consume_waiters
    // in task_group.cpp to guarantee visibility of `interrupted'.
    // 设置bthread当前等待的butex，然后当前的bthread会休眠，直到被唤醒才会继续执行
    bbw.task_meta->current_waiter.store(&bbw, butil::memory_order_release);
    g->set_remained(wait_for_butex, &bbw);
    TaskGroup::sched(&g);

    // erase_from_butex_and_wakeup (called by TimerThread) is possibly still
    // running and using bbw. The chance is small, just spin until it's done.
    // bthread被唤醒后会从这里继续执行，如果是被TimerThread唤醒，则有可能 erase_from_butex_and_wakeup 还没有执行完成，这里等待一会儿
    BT_LOOP_WHEN(unsleep_if_necessary(&bbw, get_global_timer_thread()) < 0,
                 30/*nops before sched_yield*/);
    
    // If current_waiter is NULL, TaskGroup::interrupt() is running and using bbw.
    // Spin until current_waiter != NULL.
    BT_LOOP_WHEN(bbw.task_meta->current_waiter.exchange(
                     NULL, butil::memory_order_acquire) == NULL,
                 30/*nops before sched_yield*/);
#ifdef SHOW_BTHREAD_BUTEX_WAITER_COUNT_IN_VARS
    num_waiters << -1;
#endif

    bool is_interrupted = false;
    if (bbw.task_meta->interrupted) {
        // Race with set and may consume multiple interruptions, which are OK.
        bbw.task_meta->interrupted = false;
        is_interrupted = true;
    }
    // If timed out as well as value unmatched, return ETIMEDOUT.
    if (WAITER_STATE_TIMEDOUT == bbw.waiter_state) {
        errno = ETIMEDOUT;
        return -1;
    } else if (WAITER_STATE_UNMATCHEDVALUE == bbw.waiter_state) {
        errno = EWOULDBLOCK;
        return -1;
    } else if (is_interrupted) {
        errno = EINTR;
        return -1;
    }
    return 0;
}
```
## 唤醒操作
1. butex_wake唤醒等待的一个bthread
2. butex_wake_all 唤醒所有等待的bthread，整体功能与butex_wake相似
3. butex_wake_except 唤醒除指定bthread之外所有的bthread
4. butex_requeue : 唤醒最早的一个，将剩下的移动到另一个ButexWaiter的等待列表中

```
int butex_wake(void* arg) {
    Butex* b = container_of(static_cast<butil::atomic<int>*>(arg), Butex, value);

    // step1 : 加锁，从等待链表中移除第一个等待的对象
    ButexWaiter* front = NULL;
    {
        BAIDU_SCOPED_LOCK(b->waiter_lock);
        if (b->waiters.empty()) {
            return 0;
        }
        front = b->waiters.head()->value();
        front->RemoveFromList();
        front->container.store(NULL, butil::memory_order_relaxed);
    }

    // step2 : 如果等待的对象是一个pthread，则唤醒pthread
    if (front->tid == 0) {
        wakeup_pthread(static_cast<ButexPthreadWaiter*>(front));
        return 1;
    }

    // step3 : 如果是一个bhread，则依次执行下列操作
    ButexBthreadWaiter* bbw = static_cast<ButexBthreadWaiter*>(front);
    // 检查是否需要从定时器线程中删除，在调用butex_wait时，可以使用
    // 超时参数，该参数会在定时器线程中注册等待，此处唤醒后需要从中删除
    unsleep_if_necessary(bbw, get_global_timer_thread());

    // 对于新唤醒的bthread, 马上调度执行
    TaskGroup* g = tls_task_group;
    if (g) {
        TaskGroup::exchange(&g, bbw->tid);
    } else {
        bbw->control->choose_one_group()->ready_to_run_remote(bbw->tid);
    }
    return 1;
}

int butex_requeue(void* arg, void* arg2) {
    Butex* b = container_of(static_cast<butil::atomic<int>*>(arg), Butex, value);
    Butex* m = container_of(static_cast<butil::atomic<int>*>(arg2), Butex, value);

    ButexWaiter* front = NULL;
    {
        // step1 : 将两个列表都加锁，然后从b的列表中删除，添加到m的等待列表中
        std::unique_lock<internal::FastPthreadMutex> lck1(b->waiter_lock, std::defer_lock);
        std::unique_lock<internal::FastPthreadMutex> lck2(m->waiter_lock, std::defer_lock);
        butil::double_lock(lck1, lck2);
        if (b->waiters.empty()) {
            return 0;
        }

        front = b->waiters.head()->value();
        front->RemoveFromList();
        front->container.store(NULL, butil::memory_order_relaxed);

        while (!b->waiters.empty()) {
            ButexWaiter* bw = b->waiters.head()->value();
            bw->RemoveFromList();
            m->waiters.Append(bw);
            bw->container.store(m, butil::memory_order_relaxed);
        }
    }

    // 如果唤醒的是 pthread， 则调用 wakeup_pthread
    if (front->tid == 0) {  // which is a pthread
        wakeup_pthread(static_cast<ButexPthreadWaiter*>(front));
        return 1;
    }

    // 立刻调度执行新唤醒的bthread
    ButexBthreadWaiter* bbw = static_cast<ButexBthreadWaiter*>(front);
    unsleep_if_necessary(bbw, get_global_timer_thread());
    TaskGroup* g = tls_task_group;
    if (g) {
        TaskGroup::exchange(&g, front->tid);
    } else {
        bbw->control->choose_one_group()->ready_to_run_remote(front->tid);
    }
    return 1;
}

```

