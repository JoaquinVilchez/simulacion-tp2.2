import random
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def UNIFORM (a,b):
    x=[]
    for i in range(1000):
        r = round(random.random(), 4)
        x.append(a+(b-a)*r)
    return x

def EXPENT (alfa):
    ex = 1/ alfa
    x = []
    for i in range(1000):
        r = random.random()
        x += [-ex*(math.log(r))]
    return x

def GAMMA (k,a):
    x=[]
    for i in range(1, 1000):
        tr=1.0
        for j in range(1,k):
            r = random.random()
            tr=tr*r
        x.append(-(math.log10(tr))/a)
    return x

def PASCAL(k,q):
    nx = []
    for i in range(1000):
        tr = 1
        qr = math.log10(q)
        for j in range(k):
            r = random.random()
            tr *= r
        x = int(math.log10(tr)//qr)
        nx.append(x)
    return nx

def BINOMIAL (n,p):
    x=[]
    for i in range(1000):
        y=0
        for j in range(1,n):
            r = random.random()
            if (r-p) <0:
                y+=1.0
        x.append(y)
    return x

def POISSON(p):
    listado_poisson = []
    for i in range(10000):
        x = 0
        b = np.exp(-p)
        tr = 1
        r = np.random.rand()
        tr = tr*r

        if((tr-b)>=0):
            x = x+1
            r = np.random.rand()
            tr = tr*r

        listado_poisson.append(tr-b)
    
    return listado_poisson

def HIPERGEOMETRICA(tn, ns, p):
    listado_hipergeometrica=[]
    for i in range(1000):    
        p_variable=p
        tn_variable=tn
        x=0
        s=0

        for i in range(1,ns):
            r = np.random.rand()        
            if (r-p_variable<=0):
                s=1
                x=x+1
            else:
                s=0
            p_variable=(tn_variable*p_variable-s)/(tn_variable-1)
            tn_variable=tn_variable-1
        
        listado_hipergeometrica.append(p_variable)
    
    return listado_hipergeometrica

def NORMAL(mean,sd):
    lista_normal=[]
    for j in range(1000):
        sum=0.0
        for i in (1, 12):
            r=random.random()
            sum=sum+r
        x=(sd*(sum-6.0)+mean)
        lista_normal.append(x)
    return lista_normal

def EMPIRICA():
  lista_empirica=[]
  p=[0.273,0.037,0.195,0.009,0.124,0.058,0.062,0.151,0.047,0.044]
  for j in range (1000):
      r=random.random()
      a=0
      for i in p:
        a+=i
        if (r<=a):
          break
      lista_empirica.append(a)
  return lista_empirica


Uniforme=(UNIFORM(1,3))
Gamma=(GAMMA(3, 1))
Exponencial=(EXPENT(1))
Pascal=PASCAL(3,0.3)
Binomial = BINOMIAL (1000, 0.3)
Poisson = POISSON(100)
Hipergeometrica = HIPERGEOMETRICA(10,5,0.4)
Normal=NORMAL(2.35,85.5)
Empirica=EMPIRICA()
def plotear(U, G, E, P, B, PS, H, N, EM):
    plt.title("Distribución Uniforme")
    plt.hist(U)
    plt.show()
    plt.title("Distribución Exponencial")
    plt.hist(E)
    plt.show()
    plt.title("Distribución Gamma")
    plt.hist(G)
    plt.show()
    plt.title("Distribución Pascal")
    plt.hist(P)
    plt.show()
    plt.title("Distribución Binomial")
    plt.hist(B)
    plt.show()
    plt.title("Distribución de Poisson")
    plt.hist(PS)
    plt.show()
    plt.title("Distribución Hipergeometrica")
    plt.hist(H)
    plt.show()
    plt.title("Distribucion Normal")
    plt.hist(N)
    plt.show()
    plt.title("Distribucion Empirica")
    plt.plot(EM)
    plt.show()
    
plotear(Uniforme, Exponencial, Gamma, Pascal, Binomial, Poisson, Hipergeometrica, Normal, Empirica)


