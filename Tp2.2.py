import math
def normal(x, mean, sd):
    var = float(sd) ** 2
    pi = 3.1415926
    denom = math.sqrt(2 * pi * var)
    num = math.exp(-(float(x) - float(mean)) ** 2 / (2 * var))
    return num / denom

def pascal(k,p,r):
    #p es la probabilidad
    #r es cuando quiero que suceda (hasta q...)
    # k a partir de cuando quiero que suceda
    combinacion=math.factorial(k)/(math.factorial(r)*(math.factorial(k-r)))
    num=(p**r)*((1-p)**(k-r))
    return combinacion*num

def binomial(k,n,p):
    #p es la probabilidad
    #n repeticiones independientes
    #k aca no se q es k
    combinacion=math.factorial(n)/(math.factorial(k)*math.factorial(n-k))
    num=(p**k)*((1-p)**(n-k))
    return combinacion*num
