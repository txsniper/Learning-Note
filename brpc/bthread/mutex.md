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
bthread_mutex采用了类似于futex的方式，使用一个32bit的数据指针(bthread_mutex_t.butex)
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
可以看到bthread_mutex_t.butex指向的实际上是对应的Butex.value