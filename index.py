import numpy as np
from matplotlib import pyplot as plt

#POISSON
def poisson(p):
    x = 0
    b = np.exp(-p)
    tr = 1
    r = np.random.rand()
    tr = tr*r

    if((tr-b)>=0):
        x = x+1
        r = np.random.rand()
        tr = tr*r

    return tr-b
#FIN POISSON

#DISTRIBUCION HIPERGEOMETRICA

# tn=Tamano de la poblacion
# ns=Tamano de la muestra
# p=Proporcion total de elementos de clase I

def hipergeometrica(tn, ns, p):
    x=0

    for i in range(1,ns):
        r = np.random.rand()        
        if (r-p<=0):
            s=1
            x=x+1
        else:
            s=0
        p=(tn*p-s)/(tn-1)
        tn=tn-1
    
    return p
#DISTRIBUCION HIPERGEOMETRICA


#Poisson
listado_poisson = []
for i in range(10000):
    listado_poisson.append(poisson(5*20))

print(listado_poisson)
plt.hist(listado_poisson)
plt.show()

#Hipergeometrica
listado_hipergeometrica = []
for i in range(10000):
    listado_hipergeometrica.append(hipergeometrica(10,5,0.4))
print(listado_hipergeometrica)
plt.hist(listado_hipergeometrica)
plt.show()







