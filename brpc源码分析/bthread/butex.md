## 介绍
bthread_mutex_t的底层使用butex来完成锁的功能，对于一把锁来说，主要完成两个功能：  
1. 利用状态同步不同执行流之间的操作
2. 当多个执行流对同一把锁加锁时，没有获得锁的执行流挂起等待，获得锁的执行流会在释放锁之后唤醒挂起等待的执行流
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
2. **防止创建锁的线程在销毁之后，其他线程访问锁导致Crash，这个问题会在下面的分析时详细解释**
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
## 唤醒等待的Butex
1. butex_wake唤醒等待的一个bthread
2. butex_wake_all 唤醒所有等待的bthread
3. butex_wake_except 唤醒除指定bthread之外所有的bthread

