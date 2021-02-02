import math
import numpy as np
import csv
from itertools import chain
import random

Feature_number=4
all_Feature=False
CUS_NUMBER=50
Training_number=50
count_setosa = 0
count_versicolor = 0 
count_virginica = 0

k=0
l=40
fold = []
Output = [] 
variance = []
h =np.array([0.01,0.5,10])

confussion_matrix = [[0] * 3] * 3
confussion_matrix1 = [[0] * 3] * 3
confussion_matrix2 = [[0] * 3] * 3
confussion_matrix3 = [[0] * 3] * 3
confussion_matrix4 = [[0] * 3] * 3

Iris_setosa= []
setosa_train1=[]
setosa_train2=[]
setosa_train3=[]
setosa_train4=[]
setosa_train5=[]

Iris_versicolor=[]
versicolor_train1=[]
versicolor_train2=[]
versicolor_train3=[]
versicolor_train4=[]
versicolor_train5=[]

Iris_virginica=[]
virginica_train1=[]
virginica_train2=[]
virginica_train3=[]
virginica_train4=[]
virginica_train5=[]

maximum_density = np.array([])

Iris=[]
label=["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]

def get_mean_vector(A):
    mean_vector=[]
    for i in range(Feature_number):
        sum=0
        for value in A[i]:
            sum=sum+ float(value)#accumulate all element in row i
        mean_vector.append(float(sum/len(A[i])))#add average value to MEAN_VECTOR
    return mean_vector

def norm_pdf_multivariate(x,mu,sigma):
    
    size = len(x)
    if size == len(mu) and (size, size) == sigma.shape:
        det = np.linalg.det(sigma)
        if det == 0:
            raise NameError("The covariance matrix can't be singular")

        norm_const = 1.0/ ( math.pow((2*3.1416),float(size)/2) * math.pow(det,1.0/2) )
        zip_object = zip(x, mu)
        x_mu = []
        for list1_i, list2_i in zip_object:
             x_mu.append(list1_i-list2_i)
        
        numpy_array_x_mu = np.array(x_mu)
        inv = np.linalg.inv(sigma)
        result = math.pow(math.e, -0.5 * (numpy_array_x_mu .dot( inv ).dot( numpy_array_x_mu.T)))
        return norm_const * result
    else:
        raise NameError("The dimensions of the input don't match")
    


def data_processing():
    X=-1
    fn=open("iris.data","r")
    for row in csv.DictReader(fn,label):
        X=X+1
        for i in range(Feature_number):
            Iris.append(row[label[i]])
            if str(row["class"]) == "Iris-setosa":
                if all_Feature== True:
                    Iris_setosa.append(row[label[i]])
                else:
                    if X%(Training_number/CUS_NUMBER)==0 and len(Iris_setosa)<CUS_NUMBER*4:
                        Iris_setosa.append(row[label[i]])
            elif str(row["class"]) == "Iris-versicolor":
                if all_Feature== True:
                    Iris_versicolor.append(row[label[i]])
                else:
                    if X%(Training_number/CUS_NUMBER)==0 and len(Iris_versicolor)<CUS_NUMBER*4:
                        Iris_versicolor.append(row[label[i]])
            else:
                    if all_Feature== True:
                        Iris_virginica.append(row[label[i]])
                    else:
                        if X%(Training_number/CUS_NUMBER)==0 and len(Iris_virginica)<CUS_NUMBER*4:
                            Iris_virginica.append(row[label[i]])
    fn.close() 
    
def to_matrix(l, n):
    return [l[i:i+n] for i in range(0, len(l), n)]

def data_availablity(test_I):
       
     
     setosa = Iris_setosa
     versicolor = Iris_versicolor
     virginica = Iris_virginica
     setosa = to_matrix(setosa,4)
     #print(setosa)
     versicolor = to_matrix(versicolor,4)
     virginica = to_matrix(virginica,4)
     setosa = [[float(y) for y in x] for x in setosa]
     versicolor = [[float(y) for y in x] for x in versicolor]
     virginica = [[float(y) for y in x] for x in virginica]
     
     for i in range(50):
      if(test_I == setosa[i]):
          return 'setosa';
      elif(test_I == versicolor[i]):
          return 'versicolor'
      elif(test_I == virginica[i]): 
          return 'virginica';

def get_variance(data):
      n = len(data)
      mean = sum(data) / n
      deviations = [(x - mean) ** 2 for x in data]
      variance = sum(deviations) / n
      return variance

def generate_parzebWindow_confussion_matrix(test_data,train_data,bandWidth,mean_setosa,mean_versicolor,
                               mean_virginica,covMatrix_setosa,covMatrix_versicolor,covMatrix_virginica):
      C_matrix = np.array([[0] * 3] * 3)
      Iris_setosa_Posteriror = 0
      Iris_versicolor_Posterior = 0
      Iris_virginica_Posterior = 0
      total_setosa_Posteriror = 0
      total_versicolor_Posterior = 0
      total_virginica_Posterior = 0

      for i in range(len(test_data)):
          for j in range(len(train_data)):
               zip_object = zip(test_data[i], train_data[j])
               X_XI= []
               for list1_i, list2_i in zip_object:
                 X_XI.append(list1_i-list2_i)
              
               newList = [x / bandWidth for x in X_XI]
               Iris_setosa_Posteriror =  (norm_pdf_multivariate(newList, mean_setosa, covMatrix_setosa))
               total_setosa_Posteriror = total_setosa_Posteriror + Iris_setosa_Posteriror
               Iris_versicolor_Posterior = (norm_pdf_multivariate(newList, mean_versicolor, covMatrix_versicolor))
               total_versicolor_Posterior = total_versicolor_Posterior + Iris_versicolor_Posterior
               Iris_virginica_Posterior = norm_pdf_multivariate(newList, mean_virginica, covMatrix_virginica)
               total_virginica_Posterior = total_virginica_Posterior + Iris_virginica_Posterior
             
          setosa_density= (1/(40*pow(bandWidth, 4)))*total_setosa_Posteriror
          versicolor_dansity = (1/(40*pow(bandWidth, 4)))*total_versicolor_Posterior
          virginica_density = (1/(40*pow(bandWidth, 4)))*total_virginica_Posterior
          #print(setosa_density,versicolor_dansity,virginica_density,"\n")
          maximum_density = max(setosa_density,versicolor_dansity,virginica_density)
          #print(maximum_density)
          class_name = data_availablity(test_data[i])
          #print(class_name)
          if((class_name == 'setosa') and maximum_density == setosa_density):
                  C_matrix[0][0] = C_matrix[0][0] + 1;
          elif((class_name == 'setosa') and maximum_density == versicolor_dansity):
                  C_matrix[0][1] = C_matrix[0][1] + 1;
          elif((class_name== 'setosa') and maximum_density == virginica_density):
                  C_matrix[0][2]= C_matrix[0][2] + 1;
          elif((class_name == 'versicolor') and maximum_density == setosa_density):
                 C_matrix[1][0] = C_matrix[1][0] + 1;
          elif((class_name == 'versicolor') and maximum_density == versicolor_dansity):
                  C_matrix[1][1]= C_matrix[1][1] + 1;  
          elif((class_name == 'versicolor') and maximum_density == virginica_density):
                  C_matrix[1][2]=C_matrix[1][2] + 1;
          elif((class_name== 'virginica') and maximum_density == setosa_density):
                  C_matrix[2][0] = C_matrix[2][0] + 1;
          elif((class_name== 'virginica') and maximum_density == versicolor_dansity):
                 C_matrix[2][1]=C_matrix[2][1] + 1;
          elif((class_name== 'virginica') and maximum_density == virginica_density):
                  C_matrix[2][2] = C_matrix[2][2] + 1; 
          #print(C_matrix)
      
      
      return  C_matrix 
  
if __name__ == "__main__":
    
    data_processing()
    
    for i in range(5): 
        fold.append(Iris_setosa[k:l])
        fold.append(Iris_versicolor[k:l]) 
        fold.append(Iris_virginica[k:l]) 
        k = l
        l= l+40 
        
    fold = list(chain.from_iterable(fold))
    a = (to_matrix(fold,4))
    
test_x = a[0:30]
train_y = a[30:150]

test_x1 = a[30:60]

train_y1 = a[0:30]

for i  in range(60,150):
    train_y1.append(a[i])

test_x2 = a[60:90]
train_y2 = a[0:60]
for i  in range(90,150):
    train_y2.append(a[i])

test_x3 = a[90:120]
train_y3 = a[0:90]
for i  in range(120,150):
    train_y3.append(a[i])

test_x4 = a[120:150]
train_y4 = a[0:120]

train_y = [[float(y) for y in x] for x in train_y]
test_x = [[float(y) for y in x] for x in test_x]

bandWidth = h[0]

random.shuffle(train_y)
random.shuffle(test_x)

for i in range(len(train_y)):
    if(data_availablity(train_y[i])== 'setosa'):
        setosa_train1.append(train_y[i])
    elif(data_availablity(train_y[i]) == 'versicolor'):
        versicolor_train1.append(train_y[i])
    else: virginica_train1.append(train_y[i])

setosa_train1 = np.array(setosa_train1)
#mean_setosa_result1 = get_mean_vector(setosa_train1)
mean_setosa_result1 = np.mean(setosa_train1, axis= 0)
variance_setosa1 = np.var(setosa_train1, axis = 0) 

covMatrix_setosa_result1 = [[0] * 4] * 4
covMatrix_setosa_result1 = [[0] * len(variance_setosa1) for _ in range(len(variance_setosa1))]
for i, e in enumerate(variance_setosa1):
    covMatrix_setosa_result1[i][i] = e

covMatrix_setosa_result1= np.array(covMatrix_setosa_result1)

versicolor_train1 = np.array(versicolor_train1)
#mean_versicolor_result1 = get_mean_vector(versicolor_train1)
mean_versicolor_result1 = np.mean(versicolor_train1,axis =0)
covMatrix_versicolor_result1 = [[0] * 4] * 4
variance_versicolor1= np.var(versicolor_train1, axis = 0)

covMatrix_versicolor_result1 = [[0] * len(variance_versicolor1) for _ in range(len(variance_versicolor1))]
for i, e in enumerate(variance_versicolor1):
    covMatrix_versicolor_result1[i][i] = e
    
covMatrix_versicolor_result1 = np.array(covMatrix_versicolor_result1)

virginica_train1 = np.array(virginica_train1)
#mean_virginica_result1 = get_mean_vector(virginica_train1)
mean_virginica_result1 = np.mean(virginica_train1, axis=0)

covMatrix_virginica_result1= [[0] * 4] * 4
variance_virginica1= np.var(virginica_train1, axis = 0)

covMatrix_virginica_result1 = [[0] * len(variance_virginica1) for _ in range(len(variance_virginica1))]
for i, e in enumerate(variance_virginica1):
    covMatrix_virginica_result1[i][i] = e
    
covMatrix_virginica_result1 = np.array(covMatrix_virginica_result1)

confussion_matrix = np.array(generate_parzebWindow_confussion_matrix(test_x,train_y,bandWidth,mean_setosa_result1,mean_versicolor_result1,
                                                        mean_virginica_result1,covMatrix_setosa_result1,
                                                        covMatrix_versicolor_result1,covMatrix_virginica_result1))

Sum1 = np.trace(confussion_matrix)
Accuracy1 = (Sum1/30)*100
variance.append(Accuracy1)
print("confussion matrix of first fold:\n",confussion_matrix,"\n")
print("Accuracy of First fold:", Accuracy1,"%")

train_y1 = [[float(y) for y in x] for x in train_y1]
test_x1 = [[float(y) for y in x] for x in test_x1]

random.shuffle(train_y1)
random.shuffle(test_x1)

for i in range(len(train_y1)):
    if(data_availablity(train_y1[i])== 'setosa'):
        setosa_train2.append(train_y1[i])
    elif(data_availablity(train_y1[i]) == 'versicolor'):
        versicolor_train2.append(train_y1[i])
    else: virginica_train2.append(train_y1[i])

setosa_train2 = np.array(setosa_train2)
#mean_setosa_result2 = get_mean_vector(setosa_train2)
mean_setosa_result2 = np.mean(setosa_train2, axis =0)
variance_setosa2 = np.var(setosa_train2, axis = 0) 

covMatrix_setosa_result2 = [[0] * 4] * 4
covMatrix_setosa_result2 = [[0] * len(variance_setosa2) for _ in range(len(variance_setosa2))]
for i, e in enumerate(variance_setosa2):
    covMatrix_setosa_result2[i][i] = e

covMatrix_setosa_result2= np.array(covMatrix_setosa_result2)

versicolor_train2 = np.array(versicolor_train2)
#mean_versicolor_result2 = get_mean_vector(versicolor_train2)
mean_versicolor_result2 = np.mean(versicolor_train2, axis=0)
covMatrix_versicolor_result2 = [[0] * 4] * 4
variance_versicolor2= np.var(versicolor_train2, axis = 0)

covMatrix_versicolor_result2 = [[0] * len(variance_versicolor2) for _ in range(len(variance_versicolor2))]
for i, e in enumerate(variance_versicolor2):
    covMatrix_versicolor_result2[i][i] = e
    
covMatrix_versicolor_result2 = np.array(covMatrix_versicolor_result2)

virginica_train2 = np.array(virginica_train2)
#mean_virginica_result2 = get_mean_vector(virginica_train2)
mean_virginica_result2 = np.mean(virginica_train2, axis=0)

covMatrix_virginica_result2= [[0] * 4] * 4
variance_virginica2= np.var(virginica_train2, axis = 0)

covMatrix_virginica_result2 = [[0] * len(variance_virginica2) for _ in range(len(variance_virginica2))]
for i, e in enumerate(variance_virginica2):
    covMatrix_virginica_result2[i][i] = e
    
covMatrix_virginica_result2 = np.array(covMatrix_virginica_result2)

confussion_matrix1 = np.array(generate_parzebWindow_confussion_matrix(test_x1,train_y1,bandWidth,mean_setosa_result2,mean_versicolor_result2,
                                                       mean_virginica_result2,covMatrix_setosa_result2,
                                                       covMatrix_versicolor_result2,covMatrix_virginica_result2))

Sum2 = np.trace(confussion_matrix1)
Accuracy2 = (Sum2/30)*100
variance.append(Accuracy2)
print("confussion matrix of second fold:\n",confussion_matrix1,"\n")
print("Accuracy of 2nd fold:", Accuracy2,"%")

train_y2 = [[float(y) for y in x] for x in train_y2]
test_x2 = [[float(y) for y in x] for x in test_x2]

random.shuffle(train_y2)
random.shuffle(test_x2)

for i in range(len(train_y2)):
    if(data_availablity(train_y2[i])== 'setosa'):
        setosa_train3.append(train_y2[i])
    elif(data_availablity(train_y2[i]) == 'versicolor'):
        versicolor_train3.append(train_y2[i])
    else: virginica_train3.append(train_y2[i])

setosa_train3 = np.array(setosa_train3)
#mean_setosa_result3 = get_mean_vector(setosa_train3)
mean_setosa_result3 = np.mean(setosa_train3, axis=0)
variance_setosa3 = np.var(setosa_train3, axis = 0) 

covMatrix_setosa_result3 = [[0] * 4] * 4
covMatrix_setosa_result3 = [[0] * len(variance_setosa3) for _ in range(len(variance_setosa3))]
for i, e in enumerate(variance_setosa3):
    covMatrix_setosa_result3[i][i] = e

covMatrix_setosa_result3 = np.array(covMatrix_setosa_result3)

versicolor_train3 = np.array(versicolor_train3)
#mean_versicolor_result3 = get_mean_vector(versicolor_train3)
mean_versicolor_result3 = np.mean(versicolor_train3, axis =0)
covMatrix_versicolor_result3 = [[0] * 4] * 4
variance_versicolor3= np.var(versicolor_train3, axis = 0)

covMatrix_versicolor_result3 = [[0] * len(variance_versicolor3) for _ in range(len(variance_versicolor3))]
for i, e in enumerate(variance_versicolor3):
    covMatrix_versicolor_result3[i][i] = e
    
covMatrix_versicolor_result3 = np.array(covMatrix_versicolor_result3)

virginica_train3 = np.array(virginica_train3)
#mean_virginica_result3 = get_mean_vector(virginica_train3)
mean_virginica_result3 = np.mean(virginica_train3, axis=0)

covMatrix_virginica_result3= [[0] * 4] * 4
variance_virginica3= np.var(virginica_train3, axis = 0)

covMatrix_virginica_result3 = [[0] * len(variance_virginica3) for _ in range(len(variance_virginica3))]
for i, e in enumerate(variance_virginica3):
    covMatrix_virginica_result3[i][i] = e
    
covMatrix_virginica_result3 = np.array(covMatrix_virginica_result3)

confussion_matrix2 = np.array(generate_parzebWindow_confussion_matrix(test_x2,train_y2,bandWidth,mean_setosa_result3,mean_versicolor_result3,
                                                       mean_virginica_result3,covMatrix_setosa_result3,
                                                        covMatrix_versicolor_result3,covMatrix_virginica_result3))
Sum3 = np.trace(confussion_matrix2)
Accuracy3 = (Sum3/30)*100
variance.append(Accuracy3)
print("confussion matrix of 3rd fold:\n",confussion_matrix2,"\n")
print("Accuracy of 3rd fold:", Accuracy3,"%")

train_y3 = [[float(y) for y in x] for x in train_y3]
test_x3 = [[float(y) for y in x] for x in test_x3]

random.shuffle(train_y3)
random.shuffle(test_x3)

for i in range(len(train_y3)):
    if(data_availablity(train_y3[i])== 'setosa'):
        setosa_train4.append(train_y3[i])
    elif(data_availablity(train_y3[i]) == 'versicolor'):
        versicolor_train4.append(train_y3[i])
    else: virginica_train4.append(train_y3[i])

setosa_train4 = np.array(setosa_train4)
#mean_setosa_result4 = get_mean_vector(setosa_train4)
mean_setosa_result4 = np.mean(setosa_train4, axis =0)
variance_setosa4 = np.var(setosa_train4, axis = 0) 

covMatrix_setosa_result4 = [[0] * 4] * 4
covMatrix_setosa_result4 = [[0] * len(variance_setosa4) for _ in range(len(variance_setosa4))]
for i, e in enumerate(variance_setosa4):
    covMatrix_setosa_result4[i][i] = e

covMatrix_setosa_result4 = np.array(covMatrix_setosa_result4)

versicolor_train4 = np.array(versicolor_train4)
#mean_versicolor_result4 = get_mean_vector(versicolor_train4)
mean_versicolor_result4 = np.mean(versicolor_train4, axis =0)
covMatrix_versicolor_result4 = [[0] * 4] * 4
variance_versicolor4= np.var(versicolor_train4, axis = 0)

covMatrix_versicolor_result4 = [[0] * len(variance_versicolor4) for _ in range(len(variance_versicolor4))]
for i, e in enumerate(variance_versicolor4):
    covMatrix_versicolor_result4[i][i] = e
    
covMatrix_versicolor_result4 = np.array(covMatrix_versicolor_result4)

virginica_train4 = np.array(virginica_train4)
#mean_virginica_result4 = get_mean_vector(virginica_train4)
mean_virginica_result4 = np.mean(virginica_train4, axis=0)

covMatrix_virginica_result4= [[0] * 4] * 4
variance_virginica4= np.var(virginica_train4, axis = 0)

covMatrix_virginica_result4 = [[0] * len(variance_virginica4) for _ in range(len(variance_virginica4))]
for i, e in enumerate(variance_virginica4):
    covMatrix_virginica_result4[i][i] = e
    
covMatrix_virginica_result4 = np.array(covMatrix_virginica_result4)

confussion_matrix3 = np.array(generate_parzebWindow_confussion_matrix(test_x3,train_y3,bandWidth,mean_setosa_result4,mean_versicolor_result4,
                                                        mean_virginica_result4,covMatrix_setosa_result4,
                                                        covMatrix_versicolor_result4,covMatrix_virginica_result4))

Sum4 = np.trace(confussion_matrix3)
Accuracy4 = (Sum4/30)*100
variance.append(Accuracy4)
print("confussion matrix of 4th fold:\n",confussion_matrix3,"\n")
print("Accuracy of 4th fold:", Accuracy4,"%")

train_y4 = [[float(y) for y in x] for x in train_y4]
test_x4 = [[float(y) for y in x] for x in test_x4]

random.shuffle(train_y4)
random.shuffle(test_x4)

for i in range(len(train_y4)):
    if(data_availablity(train_y4[i])== 'setosa'):
        setosa_train5.append(train_y4[i])
    elif(data_availablity(train_y4[i]) == 'versicolor'):
        versicolor_train5.append(train_y4[i])
    else: virginica_train5.append(train_y4[i])

setosa_train5 = np.array(setosa_train5)
#mean_setosa_result5 = get_mean_vector(setosa_train5)
mean_setosa_result5 = np.mean(setosa_train5, axis=0)
variance_setosa5 = np.var(setosa_train5, axis = 0) 

covMatrix_setosa_result5 = [[0] * 4] * 4
covMatrix_setosa_result5 = [[0] * len(variance_setosa5) for _ in range(len(variance_setosa5))]
for i, e in enumerate(variance_setosa5):
    covMatrix_setosa_result5[i][i] = e

covMatrix_setosa_result5 = np.array(covMatrix_setosa_result5)

versicolor_train5 = np.array(versicolor_train5)
#mean_versicolor_result5 = get_mean_vector(versicolor_train5)
mean_versicolor_result5 = np.mean(versicolor_train5, axis =0)
covMatrix_versicolor_result5 = [[0] * 4] * 4
variance_versicolor5= np.var(versicolor_train5, axis = 0)

covMatrix_versicolor_result5 = [[0] * len(variance_versicolor5) for _ in range(len(variance_versicolor4))]
for i, e in enumerate(variance_versicolor4):
    covMatrix_versicolor_result5[i][i] = e
    
covMatrix_versicolor_result5 = np.array(covMatrix_versicolor_result5)

virginica_train5 = np.array(virginica_train5)
#mean_virginica_result5 = get_mean_vector(virginica_train5)
mean_virginica_result5 = np.mean(virginica_train5, axis=0)

covMatrix_virginica_result5= [[0] * 4] * 4
variance_virginica5= np.var(virginica_train5, axis = 0)

covMatrix_virginica_result5 = [[0] * len(variance_virginica5) for _ in range(len(variance_virginica5))]
for i, e in enumerate(variance_virginica5):
    covMatrix_virginica_result5[i][i] = e
    
covMatrix_virginica_result5 = np.array(covMatrix_virginica_result5)

confussion_matrix4 = np.array(generate_parzebWindow_confussion_matrix(test_x4,train_y4,bandWidth,mean_setosa_result5,mean_versicolor_result5,
                                                        mean_virginica_result5,covMatrix_setosa_result5,
                                                        covMatrix_versicolor_result5,covMatrix_virginica_result5))
Sum5 = np.trace(confussion_matrix4)
Accuracy5 = (Sum5/30)*100
variance.append(Accuracy5)
print("confussion matrix of 5th fold:\n",confussion_matrix4,"\n")
print("Accuracy of 5th fold:", Accuracy5,"% \n")

Avg_accuracy = (Accuracy1+Accuracy2+Accuracy3+Accuracy4+Accuracy5)/5
Variance = get_variance(variance)
print("Avarage accuracy:",Avg_accuracy,"%" "with variance:",Variance)

















    