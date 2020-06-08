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

Uniforme=(UNIFORM(1,3))
Gamma=(GAMMA(3, 1))
Exponencial=(EXPENT(1))
Pascal=PASCAL(3,0.3)
Binomial = BINOMIAL (1000, 0.3)
Poisson = POISSON(100)
Hipergeometrica = HIPERGEOMETRICA(10,5,0.4)

def plotear(U, G, E, P, B, PS, H):
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

plotear(Uniforme , Exponencial , Gamma , Pascal, Binomial, Poisson, Hipergeometrica)


