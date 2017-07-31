## 带引用计数的String
来源: https://github.com/vczh-libraries/Vlpp


### 数据结构字段
```
typedef unsigned __int64        vuint64_t;
typedef vint64_t                vint;

class Object
{
public:
    virtual ~Object() {

    }
};

template<typename T>
class ObjectString : public Object
{
private:
	static const T              zero;               // 领字符串的默认值

	mutable T*                  buffer;             // 缓冲区
	mutable volatile vint*      counter;            // 引用计数，指针类型
	mutable vint                start;              // 开始下标
	mutable vint                length;             // 字符串长度
	mutable vint                realLength;         // 缓冲区长度

    // 字符串以'\0'结尾
    static vint CalculateLength(const T* buffer)
    {
        vint result=0;
        while(*buffer++)result++;
        return result;
    }
......
}


```
被mutable修饰的字段可以在 const 方法中修改。

### 方法
1. 引用计数的增减
```
void Inc()const
{
	if(counter)
	{
	    INCRC(counter);
	}
}

void Dec()const
{
	if(counter)
	{
		if(DECRC(counter)==0)
		{
			delete[] buffer;
			delete counter;
		}
	}
}
```

2. 重要的函数
```
ObjectString()
{
    buffer=(T*)&zero;
    counter=0;
    start=0;
    length=0;
    realLength=0;
}

// copy constructor 
ObjectString(const ObjectString<T>& string)
{
    buffer=string.buffer;
    counter=string.counter;         // 指向相同的引用计数指针
    start=string.start;
    length=string.length;
    realLength=string.realLength;
    Inc();                          // 引用计数加一
}

// move constructor
ObjectString(ObjectString<T>&& string)
{
    // 新字符串赋值
    buffer=string.buffer;
    counter=string.counter;
    start=string.start;
    length=string.length;
    realLength=string.realLength;
	
    // 旧字符串置为空
    string.buffer=(T*)&zero;
    string.counter=0;
    string.start=0;
    string.length=0;
    string.realLength=0;
}

// 析构函数
~ObjectString()
{
    Dec();
}

// assginment orerator 
ObjectString<T>& operator=(const ObjectString<T>& string)
{
    if(this!=&string)
    {
    	Dec();
    	buffer=string.buffer;
    	counter=string.counter;
    	start=string.start;
    	length=string.length;
    	realLength=string.realLength;
    	Inc();
    }
    return *this;
}

// move assignment operator

ObjectString<T>& operator=(ObjectString<T>&& string)
{       
    if(this!=&string)
    {
        Dec();
        buffer=string.buffer;
        counter=string.counter;
        start=string.start;
        length=string.length;
        realLength=string.realLength;
			
        string.buffer=(T*)&zero;
        string.counter=0;
        string.start=0;
        string.length=0;
        string.realLength=0;
	}
    return *this;
}
```