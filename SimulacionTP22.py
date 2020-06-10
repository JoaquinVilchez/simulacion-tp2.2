import random
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.stats as ss

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
        sum=0
        for i in range(1, 12):
            r=random.random()
            sum=sum+r
        x=(sd*(sum-6))+mean
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
Binomial = BINOMIAL (30, 0.4)
Poisson = POISSON(100)
Hipergeometrica = HIPERGEOMETRICA(10,5,0.4)
Normal=NORMAL(2.35,30)
Empirica=EMPIRICA()
def plotear(U, E, G, N, P, B, PS, H, EM):
    # distribuciones continuas
    # -------------Graficar uniforme---------------
    numerosUniformes = ss.uniform.rvs(size=1000, loc = 1, scale=3)
    sns.kdeplot(numerosUniformes, label="Distribución esperada")
    sns.distplot(U, hist_kws=dict(edgecolor="k"), label="Distribución observada")
    plt.title("Distribución Uniforme")
    plt.legend(loc="upper left")
    plt.show()

    # -------------Graficar Exponencial---------------
    numerosExponenciales=ss.expon.rvs(size=1000, loc=0, scale=1)
    sns.kdeplot(numerosExponenciales, label="Distribución esperada")
    sns.distplot(E, hist_kws=dict(edgecolor="k"), label="Distribución observada")
    plt.title("Distribución Exponencial")
    plt.legend(loc="upper left")
    plt.show()

    # -------------Graficar Gamma---------------
    plt.title("Distribución Gamma")
    plt.hist(G, alpha=1, edgecolor='black')
    plt.show()

    # -------------Graficar Normal---------------
    numerosNormales = ss.norm.rvs(size=1000, loc=2.35, scale=30)
    sns.kdeplot(numerosNormales, label="Distribución esperada")
    sns.distplot(N, hist_kws=dict(edgecolor="k"), label="Distribución observada")    
    plt.title("Distribución Normal")
    plt.legend(loc="upper left")
    plt.show()
    
    # distribuciones discretas

    # -------------Graficar Pascal---------------
    plt.title("Distribución Pascal")
    plt.hist(P, alpha=1, edgecolor='black')
    plt.show()

    #------------ Graficar binomial--------------
    N, p = 30, 0.4  # parametros de forma
    binomial = ss.binom(N, p)  # Distribución
    x = np.arange(binomial.ppf(0.01),
                  binomial.ppf(0.99))
    fmp = binomial.pmf(x)  # Función de Masa de Probabilidad
    plt.plot(x, fmp, '--' , label="Distribución esperada")
    sns.distplot(B,  hist_kws=dict(edgecolor="k"), label="Distribución observada")
    plt.title("Distribución Binomial")
    plt.ylabel('probabilidad')
    plt.xlabel('valores')
    plt.legend(loc="upper left")
    plt.show()

    # -------------Graficar Poisson---------------
    numerosPoisson = ss.poisson.rvs(np.exp(-100),size=1000, loc=0)
    sns.kdeplot(numerosPoisson, label="Distribución esperada")
    sns.distplot(PS, hist_kws=dict(edgecolor="k"), label="Distribución observada")   
    plt.title("Distribución de Poisson")
    plt.legend(loc="upper left")
    plt.show()

    # -------------Graficar Hiper---------------
    plt.title("Distribución Hipergeometrica")
    plt.hist(H, alpha=1, edgecolor = 'black')
    plt.show()
    #-------------Graficar empirica------------
    plt.title("Distribucion Empirica")
    plt.plot(EM)
    plt.show()
    
plotear(Uniforme , Exponencial , Gamma , Normal, Pascal, Binomial, Poisson, Hipergeometrica, Empirica)



