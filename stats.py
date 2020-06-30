import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

'''
fig1 = plt.figure(figsize=(10, 6))

x1 = np.random.randn(1, 20)
y1 = np.random.randn(1, 20)

x2 = np.random.randn(1, 20) + 5
y2 = np.random.randn(1, 20)

x3 = np.random.randn(1, 20)
y3 = np.random.randn(1, 20) + 10

x4 = np.random.randn(1, 20) + 5
y4 = np.random.randn(1, 20) + 10

c1 = plt.plot(x1, y1, 'r.')
c2 = plt.plot(x2, y2, 'b.')
c3 = plt.plot(x3, y3, 'g.')
c4 = plt.plot(x4, y4, 'y.')

plt.show()


fig2 = plt.figure(figsize=(10, 6))
x = np.arange(0, 100, 4)
y = 5 * x + 4
r =  (np.random.randn(25, 1) * 7)
y_ = y + r.reshape((25, ))

line = plt.plot(x, y, '-')
dots = plt.plot(x, y_, 'r.')

plt.show()


def dummy_train(t):
    return (1-np.exp(-5*x))

def dummy_test(t):
    return (np.sin(2*t)/1.2-np.exp(-5-5*t))

x = np.arange(0, 1, 0.01)

fig3 = plt.figure(figsize=(6, 6))
ax = fig3.add_subplot(111)
ax.plot(x, dummy_train(x), 'b', label='Training Accuracy')
ax.plot(x, dummy_test(x), 'g--', label='Testing Accuracy')
leg = ax.legend(frameon=False, loc=4)
ax.set_xlim(0, 1)
ax.set_xticks([])
ax.set_xlabel('Epochs')
ax.set_ylabel('Accuracy')
plt.show()
'''


x = np.random.rand(30)
y = (x * 8) + np.random.rand(30)

mean = [np.mean(y) for i in range(len(x))]

fig = plt.figure(figsize=(10, 6))
sns.regplot(x, y, label='Data')
plt.plot(x, mean, 'k-', label='Mean')
plt.legend()
plt.show()
