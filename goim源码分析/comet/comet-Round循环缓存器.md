### 一. Round
Round提供了对资源的逻辑封装，它并不是一个简单的资源池，而是包含了多个资源池，当新建Channel的时候，会根据计数来返回对应的资源池，**注意，多个连接可以使用相同的资源池，Round的作用是作为资源池的分配器。**

```
type RoundOptions struct {
	Timer        int        // 定时器管理器的数量
	TimerSize    int        // 每一个定时器管理器中定时器的数量(初始化)
	Reader       int        // Reader bytes.Pool的数量
	ReadBuf      int        // Reader bytes.Pool中内存块的数量
	ReadBufSize  int        // Reader bytes.Pool中内存块的大小
	Writer       int        // Writer bytes.Pool的数量
	WriteBuf     int        // Writer bytes.Pool中内存块的数量
	WriteBufSize int        // Writer bytes.Pool中内存块的大小
}

// Ronnd userd for connection round-robin get a reader/writer/timer for split big lock.
type Round struct {
	readers   []bytes.Pool
	writers   []bytes.Pool
	timers    []time.Timer
	options   RoundOptions
	readerIdx int
	writerIdx int
	timerIdx  int
}

// NewRound new a round struct.
func NewRound(options RoundOptions) (r *Round) {
	var i int
	r = new(Round)
	r.options = options
	
	// reader
	// 初始化多个读缓存池
	r.readers = make([]bytes.Pool, options.Reader)
	for i = 0; i < options.Reader; i++ {
		r.readers[i].Init(options.ReadBuf, options.ReadBufSize)
	}
	
	// writer
	// 初始化多个写缓存池
	r.writers = make([]bytes.Pool, options.Writer)
	for i = 0; i < options.Writer; i++ {
		r.writers[i].Init(options.WriteBuf, options.WriteBufSize)
	}
	
	// timer
	// 初始化多个定时器管理器
	r.timers = make([]time.Timer, options.Timer)
	for i = 0; i < options.Timer; i++ {
		r.timers[i].Init(options.TimerSize)
	}
	return
}
```
### 二.获取资源
通过计数取模方式获取资源池，平衡每个资源池的使用。
```
// Timer get a timer.
func (r *Round) Timer(rn int) *time.Timer {
	return &(r.timers[rn%r.options.Timer])
}

// Reader get a reader memory buffer.
func (r *Round) Reader(rn int) *bytes.Pool {
	return &(r.readers[rn%r.options.Reader])
}

// Writer get a writer memory buffer pool.
func (r *Round) Writer(rn int) *bytes.Pool {
	return &(r.writers[rn%r.options.Writer])
}
```
