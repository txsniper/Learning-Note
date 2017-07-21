## 协程执行流程
这里以example_specific的执行流程为例解释一下协程是如何切换的，**例子中的协程都是非共享栈**。  
example_specific测试了协程私有变量(**类似于线程私有变量**)

## 背景说明
### 1. 协程是怎样执行的
每一个系统线程是一组协程的管理者，这一组协程的执行都是由这个线程在各个协程间切换实现的(**不考虑工作密取模式**)，类比到操作系统的概念，这里的线程相当于操作系统的CPU，而这里的协程则相当于操作系统的线程。
### 2. 协程如何调度
上面说了每一个系统线程管理着一组协程，为了对这组协程进行调度，线程在初始化的时候会创建一个**主协程**，它不执行任何业务代码，它的主要作用就是启动业务协程，在一个业务协程执行完毕时调度其他的业务协程执行。  
从这里我们可以看出整体的执行流程是:  
主协程------->业务协程A------->主协程-------->业务协程B  
业务协程A和业务协程B之间的执行依赖于主协程的调度
### 3. 如何处理定时器和阻塞函数
在协程的执行函数中，有时我们会sleep一段时间或者调用一些阻塞函数(比如IO函数，read/write)，如果我们不进行处理，那么会直接阻塞协程所在的线程，为了处理这种情况，我们需要对这些函数进行Hook，提供对应的非阻塞的函数。  
IO函数实质上是对文件描述符的读写操作，sleep则是等待一段时间，然后超时。可同时等待多个文件描述符的读写操作，同时还包含超时，**使用 epoll !!!**

## 整体流程
### 功能函数

#### 新建协程
如果当前线程没有创建协程执行环境，则初始化协程执行环境，同时创建线程的主协程，**主协程充当调度者的角色**
```
// 新建协程
int co_create( stCoRoutine_t **ppco,const stCoRoutineAttr_t *attr,pfn_co_routine_t pfn,void *arg )
{
	// step1 : 如果当前线程还没有初始化执行环境，则首先进行初始化
	if( !co_get_curr_thread_env() ) 
	{
		co_init_curr_thread_env();
	}

	// step2 : 创建协程
	stCoRoutine_t *co = co_create_env( co_get_curr_thread_env(), attr, pfn,arg );
	*ppco = co;
	return 0;
}
```

#### 协程切换
两种切换方式：
1. 主动切换到某个指定的协程(主协程常用来主动切换到业务协程)
2. 主动让出执行线程，业务协程执行完毕后会主动让出执行。
```
// 切换到 co 协程
void co_resume( stCoRoutine_t *co )
{
	stCoRoutineEnv_t *env = co->env;
	// 当前执行的协程
	stCoRoutine_t *lpCurrRoutine = env->pCallStack[ env->iCallStackSize - 1 ];
	if( !co->cStart )
	{
		// 配置寄存器和函数栈, CoRoutineFunc 配置为栈中返回函数, co配置为返回函数的参数
		coctx_make( &co->ctx,(coctx_pfn_t)CoRoutineFunc,co,0 );
		co->cStart = 1;
	}
	env->pCallStack[ env->iCallStackSize++ ] = co;
	co_swap( lpCurrRoutine, co );
}

void co_yield_env( stCoRoutineEnv_t *env )
{
	// 从pCallStack中取出下一个执行的协程，进行切换
	stCoRoutine_t *last = env->pCallStack[ env->iCallStackSize - 2 ];
	stCoRoutine_t *curr = env->pCallStack[ env->iCallStackSize - 1 ];
	env->iCallStackSize--;
	co_swap( curr, last);
}
```
对于每一个执行线程，有一个协程栈 pCallStack (不是协程执行函数栈)，栈中包含了当前加入到执行队列中的协程，这里与另外的协程库(libgo，百度的bthread)不同的是，libco的执行线程并不包含一个待执行的协程列表，这里只有执行co_resume切换到目标协程时才将协程加入到pCallStack(所以所有example中都是利用co_create新建一个协程，然后马上执行co_resume切换到它)。 **注意，我们在初始化线程的协程执行环境时，会将主协程添加到 pCallStack[0]的位置，也就是说，当pCallStack中所有业务协程执行完之后，将一直执行主协程**   
libco中为了切换到目标协程，手动的设置了函数栈的返回地址：

`coctx_make( &co->ctx,(coctx_pfn_t)CoRoutineFunc,co,0 )`  
coctx_make配置了协程的执行上下文，主要是将CoRoutineFunc设置为栈返回函数，将co设置为参数。
CoRoutineFunc就是每个协程执行的函数，
```
static int CoRoutineFunc( stCoRoutine_t *co,void * )
{
	// 执行协程任务
	if( co->pfn )
	{
		co->pfn( co->arg );
	}
	co->cEnd = 1;

	stCoRoutineEnv_t *env = co->env;

	// 执行完毕，切换
	co_yield_env( env );

	return 0;
}
```

#### 切换内部流程

co_swap是流程切换的内部函数，**该函数以执行coctx_swap为界，coctx_swap之前的代码是当前协程本次执行，coctx_swap之后的代码则是在执行了pending_co之后切换回当前协程执行。**
```
// 切换协程
void co_swap(stCoRoutine_t* curr, stCoRoutine_t* pending_co)
{
 	stCoRoutineEnv_t* env = co_get_curr_thread_env();

	//get curr stack sp
	char c;
	curr->stack_sp= &c;

	// 独立栈，不用设置
	if (!pending_co->cIsShareStack)
	{
		env->pending_co = NULL;
		env->occupy_co = NULL;
	}
	else 
	{
		// 共享栈，需要保存当前的栈内容
		env->pending_co = pending_co;
		//get last occupy co on the same stack mem
		// 获取之前占有栈空间的协程
		stCoRoutine_t* occupy_co = pending_co->stack_mem->occupy_co;
		//set pending co to occupy thest stack mem;
		pending_co->stack_mem->occupy_co = pending_co;

		// 保存之前占用栈空间的协程
		env->occupy_co = occupy_co;

		// 将之前占用栈空间的协程栈数据保存下来
		if (occupy_co && occupy_co != pending_co)
		{
			save_stack_buffer(occupy_co);
		}
	}

	//swap context
	// 最重要的函数：切换执行上下文
	coctx_swap(&(curr->ctx),&(pending_co->ctx) );

	// 切换回来(co_resume实现)，恢复栈内容
	//stack buffer may be overwrite, so get again;
	stCoRoutineEnv_t* curr_env = co_get_curr_thread_env();
	stCoRoutine_t* update_occupy_co =  curr_env->occupy_co;
	stCoRoutine_t* update_pending_co = curr_env->pending_co;
	
	if (update_occupy_co && update_pending_co && update_occupy_co != update_pending_co)
	{
		//resume stack buffer
		if (update_pending_co->save_buffer && update_pending_co->save_size > 0)
		{
			memcpy(update_pending_co->stack_sp, update_pending_co->save_buffer, update_pending_co->save_size);
		}
	}
}
```

### example_specific执行流程
首先创建10个协程，在每创建完一个协程时都立刻调用co_resume切换到新创建的协程

```
int main()
{
	stRoutineArgs_t args[10];
	for (int i = 0; i < 10; i++)
	{
		args[i].routine_id = i;
		co_create(&args[i].co, NULL, RoutineFunc, (void*)&args[i]);
		co_resume(args[i].co);
	}
	co_eventloop(co_get_epoll_ct(), NULL, NULL);
	return 0;
}
```

协程执行的业务函数 RoutineFunc

```
void* RoutineFunc(void* args)
{
	co_enable_hook_sys();
	stRoutineArgs_t* routine_args = (stRoutineArgs_t*)args;
	__routine->idx = routine_args->routine_id;
	while (true)
	{
		printf("%s:%d routine specific data idx %d\n", __func__, __LINE__, __routine->idx);
		poll(NULL, 0, 1000);
	}
	return NULL;
}
```
在业务函数循环中，不停的输出内容并睡眠1s，上面说了，为了不让poll阻塞当前线程，对poll进行了Hook，使用自己的函数代替系统poll

```
int poll(struct pollfd fds[], nfds_t nfds, int timeout)
{

	HOOK_SYS_FUNC( poll );

	// 如果没启动hoook，则直接调用系统poll
	if( !co_is_enable_sys_hook() )
	{
		return g_sys_poll_func( fds,nfds,timeout );
	}

	// 调用自己实现的poll
	return co_poll_inner( co_get_epoll_ct(),fds,nfds,timeout, g_sys_poll_func);

}
```
在 co_poll_inner函数中分了4步，前面三步将需要等待的文件描述符加入到epoll中，需要等待的超时加入到链表中，然后执行 co_yield_env 让出当前执行线程，当业务协程让出执行后，线程会执行主协程。