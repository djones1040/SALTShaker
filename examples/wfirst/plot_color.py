import numpy as np
import matplotlib.pyplot as plt

z,c = np.loadtxt('color.dat',unpack=True)
print(len(z))
c1=c[z>1.4]
c2=c[z<=1.4]
plt.hist(c1)
print(np.median(c1))
#plt.plot([np.mean(c1),np.mean(c1)],,'--')
plt.savefig('color.pdf',overwrite=True)
plt.clf()
plt.hist(c2)
print(np.median(c2))
#plt.plot([np.mean(c2),np.mean(c2)],'--')
plt.xlim((-.3,.3))
plt.savefig('color1.pdf',overwrite=True)

