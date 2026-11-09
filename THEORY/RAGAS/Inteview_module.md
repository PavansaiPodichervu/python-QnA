# Python + AI + Agentic AI Engineer Interview Preparation
## Experience Level: 6+ Years

## Batch 1: Advanced Python Interview Questions

**Target roles:** Senior Python Developer, AI Engineer, GenAI Engineer, Agentic AI Engineer

**Experience level:** 6+ years

**Question sourcing:**
- Python official documentation
- Real Python technical articles
- GitHub interview repositories
- Public interview-preparation resources

**Note:** Questions below are interview-preparation questions based on the cited technical topics. They are not claimed to be verbatim questions from a specific company's confidential interview.

---

## Q1. What is the Global Interpreter Lock (GIL) in Python? How does it affect multithreading?

**Source:** Python documentation  
https://docs.python.org/3/glossary.html#term-global-interpreter-lock

### Answer

The Global Interpreter Lock (GIL) is a mechanism in traditional CPython that allows only one thread at a time to execute Python bytecode within a process.

It simplifies memory management and protects access to Python objects.

### Example

```python
import threading

def calculate():
    total = 0

    for i in range(10_000_000):
        total += i

threads = [
    threading.Thread(target=calculate)
    for _ in range(4)
]

for thread in threads:
    thread.start()

for thread in threads:
    thread.join()
```

### Explanation

For CPU-bound Python bytecode, multiple threads generally do not provide parallel execution in traditional GIL-enabled CPython.

For I/O-bound tasks, threading can still improve performance because threads can release execution while waiting for network or file operations.

### Follow-up questions

- How does multiprocessing overcome GIL limitations?
- Does asyncio solve CPU-bound problems?
- Can Python execute threads in parallel without the GIL?

---

## Q2. What is the difference between threading, multiprocessing, and asyncio?

**Source:** Python documentation  
https://docs.python.org/3/library/threading.html  
https://docs.python.org/3/library/multiprocessing.html  
https://docs.python.org/3/library/asyncio.html

### Answer

| Feature | Threading | Multiprocessing | Asyncio |
|---|---|---|---|
| Execution | Threads | Separate processes | Cooperative tasks |
| CPU parallelism | Limited by GIL in traditional CPython | Yes | No |
| I/O concurrency | Yes | Yes | Yes |
| Memory | Shared process memory | Separate process memory | Shared event-loop process |
| Best use | Blocking I/O | CPU-intensive tasks | Async I/O |

### Example: asyncio

```python
import asyncio

async def fetch_data():
    await asyncio.sleep(2)
    return "Data received"

async def main():
    results = await asyncio.gather(
        fetch_data(),
        fetch_data(),
        fetch_data()
    )

    print(results)

asyncio.run(main())
```

### Explanation

The three operations can overlap while waiting.

Asyncio is especially useful for applications that perform many concurrent network requests, such as LLM API calls.

---

## Q3. What is the difference between a generator and an iterator?

**Source:** Python documentation  
https://docs.python.org/3/glossary.html#term-generator  
https://docs.python.org/3/glossary.html#term-iterator

### Answer

An iterator is an object implementing:

- `__iter__()`
- `__next__()`

A generator is a convenient way to create an iterator using a function containing `yield`.

### Example

```python
def generate_numbers():
    yield 10
    yield 20
    yield 30

numbers = generate_numbers()

print(next(numbers))
print(next(numbers))
print(next(numbers))
```

Output:

```text
10
20
30
```

### Explanation

Generators produce values lazily instead of storing the entire sequence in memory.

They are useful for:

- Large datasets
- Streaming responses
- Processing files
- Data pipelines

### AI engineering example

```python
def stream_chunks(chunks):
    for chunk in chunks:
        yield chunk
```

This pattern is useful when processing documents incrementally.

---

## Q4. Explain Python decorators and their practical use.

**Source:** Python documentation  
https://docs.python.org/3/glossary.html#term-decorator

### Answer

A decorator is a callable that modifies or extends the behavior of another function or class without directly changing its implementation.

### Example

```python
from functools import wraps

def logger(func):

    @wraps(func)
    def wrapper(*args, **kwargs):

        print("Function started")

        result = func(*args, **kwargs)

        print("Function completed")

        return result

    return wrapper


@logger
def generate_answer():
    return "AI response"


print(generate_answer())
```

### Output

```text
Function started
Function completed
AI response
```

### Real-world use cases

- Authentication
- Logging
- Retry mechanisms
- Timing
- Caching
- Rate limiting

### Follow-up

Why should we use `functools.wraps`?

It preserves metadata such as the original function name and documentation string.

---

## Q5. What are mutable default arguments in Python, and why are they dangerous?

**Source:** Python FAQ  
https://docs.python.org/3/faq/programming.html#why-are-default-values-shared-between-objects

### Answer

Default argument values are evaluated once when the function is defined, not every time it is called.

If the default is mutable, changes can persist across function calls.

### Incorrect implementation

```python
def add_item(item, items=[]):
    items.append(item)
    return items

print(add_item("Python"))
print(add_item("AI"))
```

Output:

```text
['Python']
['Python', 'AI']
```

### Correct implementation

```python
def add_item(item, items=None):

    if items is None:
        items = []

    items.append(item)

    return items

print(add_item("Python"))
print(add_item("AI"))
```

Output:

```text
['Python']
['AI']
```

---

## Q6. Explain Python memory management and garbage collection.

**Source:** Python documentation  
https://docs.python.org/3/c-api/memory.html  
https://docs.python.org/3/library/gc.html

### Answer

Python manages memory automatically.

Traditional CPython primarily uses:

- Reference counting
- Cyclic garbage collection

Reference counting tracks how many references point to an object.

When the reference count reaches zero, the object can generally be deallocated.

Cyclic garbage collection handles certain reference cycles that reference counting alone cannot reclaim.

### Example

```python
import gc

class Node:
    pass

a = Node()
b = Node()

a.ref = b
b.ref = a

del a
del b

gc.collect()
```

### Explanation

The two objects reference each other.

This creates a cycle.

Garbage collection can detect and reclaim unreachable cyclic objects.

---

## Q7. What is the difference between shallow copy and deep copy?

**Source:** Python documentation  
https://docs.python.org/3/library/copy.html

### Answer

A shallow copy creates a new outer object but retains references to nested objects.

A deep copy recursively copies nested objects.

### Example

```python
import copy

original = [[1, 2], [3, 4]]

shallow = copy.copy(original)

deep = copy.deepcopy(original)

original[0].append(99)

print(shallow)
print(deep)
```

Output:

```text
[[1, 2, 99], [3, 4]]
[[1, 2], [3, 4]]
```

### Practical importance

This matters when manipulating nested configuration objects, request payloads, or shared application state.

---

## Q8. Explain Python MRO.

**Source:** Python documentation  
https://docs.python.org/3/howto/mro.html

### Answer

MRO stands for Method Resolution Order.

It defines the order in which Python searches classes for methods and attributes, especially in inheritance hierarchies.

### Example

```python
class A:
    def execute(self):
        print("A")

class B(A):
    def execute(self):
        print("B")

class C(A):
    def execute(self):
        print("C")

class D(B, C):
    pass

D().execute()

print(D.mro())
```

Output:

```text
B
[<class '__main__.D'>,
 <class '__main__.B'>,
 <class '__main__.C'>,
 <class '__main__.A'>,
 <class 'object'>]
```

### Explanation

Python uses C3 linearization to determine method lookup order.

MRO is important when working with:

- Multiple inheritance
- Framework base classes
- Mixins
- Complex class hierarchies

---

## Q9. What is a context manager in Python?

**Source:** Python documentation  
https://docs.python.org/3/library/contextlib.html

### Answer

A context manager manages resources automatically using the `with` statement.

It commonly handles setup and cleanup operations.

### Example

```python
with open("data.txt", "w") as file:
    file.write("Python")
```

The file is automatically closed when execution exits the context.

### Custom context manager

```python
from contextlib import contextmanager

@contextmanager
def database_session():

    print("Opening session")

    try:
        yield
    finally:
        print("Closing session")


with database_session():
    print("Executing query")
```

### Use cases

- Database sessions
- File handling
- Locks
- Temporary resources
- Transaction management

---

## Q10. What is the difference between `__str__` and `__repr__`?

**Source:** Python documentation  
https://docs.python.org/3/reference/datamodel.html#object.__repr__

### Answer

`__str__` provides a readable representation intended for users.

`__repr__` provides a representation intended for developers and debugging.

### Example

```python
class User:

    def __init__(self, name):
        self.name = name

    def __str__(self):
        return f"User: {self.name}"

    def __repr__(self):
        return f"User(name={self.name!r})"


user = User("Alex")

print(str(user))
print(repr(user))
```

Output:

```text
User: Alex
User(name='Alex')
```

---

# Additional Advanced Python Questions

Continue preparing these questions using the same sources.

| No. | Interview Question | Source |
|---|---|---|
| 11 | What is a closure in Python? | [Python FAQ](https://docs.python.org/3/faq/programming.html) |
| 12 | Explain `*args` and `**kwargs`. | [Python Tutorial](https://docs.python.org/3/tutorial/controlflow.html#arbitrary-argument-lists) |
| 13 | What are descriptors in Python? | [Descriptor HOWTO](https://docs.python.org/3/howto/descriptor.html) |
| 14 | What is the difference between `__new__` and `__init__`? | [Data Model](https://docs.python.org/3/reference/datamodel.html) |
| 15 | How does exception handling work internally? | [Errors and Exceptions](https://docs.python.org/3/tutorial/errors.html) |
| 16 | What are Python metaclasses? | [Data Model](https://docs.python.org/3/reference/datamodel.html#metaclasses) |
| 17 | What is monkey patching? | [Python FAQ](https://docs.python.org/3/faq/programming.html) |
| 18 | What are slots and how do they affect memory? | [Data Model](https://docs.python.org/3/reference/datamodel.html#object.__slots__) |
| 19 | How do you profile Python applications? | [Profiling](https://docs.python.org/3/library/profile.html) |
| 20 | How do you optimize memory usage in Python? | [Python Memory Management](https://docs.python.org/3/c-api/memory.html) |

---

# Batch 1 Completion Checklist

- [ ] Understand GIL.
- [ ] Understand threading, multiprocessing, and asyncio.
- [ ] Understand generators and iterators.
- [ ] Understand decorators.
- [ ] Understand mutable default arguments.
- [ ] Understand memory management.
- [ ] Understand shallow and deep copy.
- [ ] Understand MRO.
- [ ] Understand context managers.
- [ ] Understand Python object representations.

---

# Source Policy for This Interview Preparation

Every batch will contain:

1. Interview question.
2. Interview-ready answer.
3. Detailed explanation.
4. Python code when applicable.
5. Practical scenario.
6. Follow-up questions.
7. Exact reference URL.
8. Source classification.

Questions will be classified as:

- **Documented topic:** directly supported by official technical documentation.
- **Published interview question:** explicitly listed in a public interview resource.
- **Derived interview question:** constructed from documented concepts and common senior engineering responsibilities.

I will not invent company attribution or claim a question was asked in an actual interview without evidence.

