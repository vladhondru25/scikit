import matplotlib.pyplot as plt

x = [i for i in range(10)]
y = [2*i for i in range(10)]

# plt.plot(x,y)
plt.scatter(x,y)
plt.xlabel('x')
plt.ylabel('y')
plt.show()