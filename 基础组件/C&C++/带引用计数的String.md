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
几个要点：**被mutable修饰的字段可以在 const 方法中修改; counter作为引用计数只有在buffer由ObjectString动态分配时才会计数，如果 buffer指向一个别的缓冲区，则counter不会增减**

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

2. 构造，析构函数
```
// 默认构造函数：构造空字符串
ObjectString()
{
    buffer=(T*)&zero;
    counter=0;
    start=0;
    length=0;
    realLength=0;
}

// copy constructor : 拷贝构造函数
ObjectString(const ObjectString<T>& string)
{
    buffer=string.buffer;
    counter=string.counter;         // 指向相同的引用计数指针
    start=string.start;
    length=string.length;
    realLength=string.realLength;
    Inc();                          // 引用计数加一
}

ObjectString(const T* _buffer, bool copy = true)
{
    CHECK_ERROR(_buffer!=0, L"ObjectString<T>::ObjectString(const T*, bool)#Cannot construct a string from nullptr.");
    if(copy)
    {
        counter=new vint(1);
        start=0;
        length=CalculateLength(_buffer);
        buffer=new T[length+1];
        memcpy(buffer, _buffer, sizeof(T)*(length+1));
            realLength=length;
	}
    else
    {
        // 外部缓冲区，引用计数指针置空
        buffer=(T*)_buffer;
        counter=0;
        start=0;
        length=CalculateLength(_buffer);
        realLength=length;
        }
}

// move constructor : 移动构造函数
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
/*
* 注意：赋值操作符的返回类型为引用
*/
// assginment orerator : 赋值操作符
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

// move assignment operator : 移动赋值操作符
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

3. 实用方法
```
// 删除字符串：
// index : 删除的位置
// count : 删除的长度 
ObjectString<T> Remove(vint index, vint count)const
{
    CHECK_ERROR(index>=0 && index<length, L"ObjectString<T>::Remove(vint, vint)#Argument index not in range.");
    CHECK_ERROR(index+count>=0 && index+count<=length, L"ObjectString<T>::Remove(vint, vint)#Argument count not in range.");
    return ObjectString<T>(*this, ObjectString<T>(), index, count);
}

// 插入字符串
// index : 插入的源字符串位置
// string : 插入的新字符串
ObjectString<T> Insert(vint index, const ObjectString<T>& string)const
{
    CHECK_ERROR(index>=0 && index<=length, L"ObjectString<T>::Insert(vint)#Argument count not in range.");
    return ObjectString<T>(*this, string, index, 0);
}

ObjectString(const ObjectString<T>& dest, 
    const ObjectString<T>& source, vint index, vint count)
{
    if(index==0 && count==dest.length && source.length==0)
    {
        buffer=(T*)&zero;
        counter=0;
        start=0;
        length=0;
        realLength=0;
    }
	else
    {
        counter=new vint(1);
        start=0;
        length=dest.length-count+source.length;
        realLength=length;
        buffer=new T[length+1];

        // 拷贝dest从 start 到 index 范围内的数据
        memcpy(buffer, dest.buffer+dest.start, sizeof(T)*index);

        // 拷贝 source 的整体数据
        memcpy(buffer+index, source.buffer+source.start, sizeof(T)*source.length);

        // 拷贝 dest 从 start + index + count 到 结尾的数据
        memcpy(buffer+index+source.length, (dest.buffer+dest.start+index+count), sizeof(T)*(dest.length-index-count));

        buffer[length]=0;
    }
}
```