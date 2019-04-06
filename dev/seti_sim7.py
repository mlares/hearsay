
# coding: utf-8

import numpy as np
import random
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import spatial as sp
import time

# NODE AND LINKED LIST CLASSES
#{{{
class Node:
    def __init__(self,initdata):
        self.data = initdata
        self.next = None

    def getData(self):
        return self.data

    def getNext(self):
        return self.next

    def setData(self,newdata):
        self.data = newdata

    def setNext(self,newnext):
        self.next = newnext


class OrderedList:
    def __init__(self):
        self.head = None

    def search(self,item):
        current = self.head
        found = False
        stop = False
        while current != None and not found and not stop:
            if current.getData() == item:
                found = True
            else:
                if current.getData() > item:
                    stop = True
                else:
                    current = current.getNext()

        return found
    
    def show(self):
        current = self.head
        while current != None:
            print current.getData()
            current = current.getNext()        
    
    def add(self,item):
        current = self.head
        previous = None
        stop = False
        while current != None and not stop:
            if current.getData() > item:
                stop = True
            else:
                previous = current
                current = current.getNext()

        temp = Node(item)
        if previous == None:
            temp.setNext(self.head)
            self.head = temp
        else:
            temp.setNext(current)
            previous.setNext(temp)
            
    def remove_first(self):
        self.head = self.head.getNext()
            

    def isEmpty(self):
        return self.head == None

    def size(self):
        current = self.head
        count = 0
        while current != None:
            count = count + 1
            current = current.getNext()

        return count
#}}}


# FIXED VARIABLES
#{{{
# radio interno de la zona galactica habitable, años luz
GHZ_inner     = 20000.  
GHZ_inner     = 0.

# radio interno de la zona galactica habitable, años luz
GHZ_outer     = 60000.  
GHZ_outer     = 100.

# tiempo medio, en años, que hay que esperar para que 
# aparezca otra CETI en la galaxia
tau_awakening = 500.  

# Tiempo medio, en años, durante el cual una CETI esta activa
tau_survive   = 1000.   

# Maxima distancia, en años luz, a la cual una CETI puede 
# enviar o recibir mensajes
D_max         = 3000.   

#}}}

# SIMULATION VARIABLES
#{{{

# Inicializar listas de datos

# lista de CETIs alguna vez activas: dictionary
CETIs = dict()

# lista de CETIs actualmente activas: ndarray
CHATs = []
CHATs_idx = []

# inicializacion del tiempo: scalar
t_now = 0

# inicializacion del ID: index
ID = 0

# lista de tiempos de eventos futuros: ordered list
t_forthcoming = OrderedList()

# maximo tiempo para simular
t_max = 100000.


status = ["","appears","dissapears",
          "sees first contact","sees last contact"]

#}}}

next_event = [0., 0, None, 1]
t_forthcoming.add(next_event)

k=0 # count number of events

# SIMULATION LOOP
#{{{
while (t_now<t_max):
    
    k=k+1 # count number of events

    t_now, ID, ID_alien, case = t_forthcoming.head.getData()

    print ''
    #print('[ case:%d | id:%d ]      <<< %f >>>' % (case, ID, t_now))
    print('[ CETI %d %s ]      <<< present time: %f >>>\n' % (ID, status[case], t_now))
        
    if case==1:
        #{{{
        #-------------------------------------
        # NEW CETI
        #-------------------------------------

        # sortear el lugar donde aparece dentro de la GHZ
        r = np.sqrt(random.random()*GHZ_outer**2 + GHZ_inner**2)
        o = random.random()*2.*np.pi
        x = r * np.cos(o)  # X position on the galactic plane
        y = r * np.sin(o)  # Y position on the galactic plane

        # sortear el tiempo de actividad
        t_active = np.random.exponential(tau_survive, 1)[0]
        t_hola = t_now
        t_chau = t_hola + t_active

        # agregar el tiempo de desaparición a la lista de tiempos
        next_event = [t_chau, ID, None, 2]
        t_forthcoming.add(next_event)

        # agregar la CETI a la lista histórica
        CETIs[ID] = (x, y, t_hola, t_chau)

        # sortear el tiempo de aparición de la próxima CETI
        # y agregar el próximo evento (tipo 1)
        t_next_awakening = np.random.exponential(tau_awakening, 1)[0]
        t_next_awakening = t_now + t_next_awakening
        next_event = [t_next_awakening, ID+1, None, 1]
        t_forthcoming.add(next_event)

        # buscar los próximos contactos con base en la CETI que se
        # acaba de crear
        try:
            tree
        except NameError:
            print 'El arbol no existe'
        else:
            # encontrar todas las CETIs dentro del radio D_max
            query_point = [x,y]  # center at the current CETI
            idx = tree.query_ball_point(query_point, r=D_max)

            print(idx)
            print('len(CHATS):%d' % len(CHATs))
            print('len(CETIS):%d' % len(CETIs))
            print(CETIs)

            for k in idx:

                print('len(CHATS):%d    k:%d' % (len(CHATs), k))

                z = np.sqrt(((np.array(query_point) - 
                              np.array(CHATs[k]))**2).sum())
            
                # Agregar a t_forthcoming los tiempos de contacto (caso 3):
                # 1.- Desde la recién agregada hacia las otras CETIs
                z = t_now + z
                next_event = [z, ID, k, 3]
                t_forthcoming.add(next_event)
                # 2.- Desde las otras CETIs hacia la recién agregada
                #t_hola = CETIs[2][2]
                t_hola = CETIs[k][2]
                t = z - t_hola
                next_event = [t, ID, k, 3]
                t_forthcoming.add(next_event)

#                # Agregar a t_forthcoming los tiempos de baja del
#                # contacto (caso 4):
#                # 1.- Desde la recién agregada hacia las otras CETIs
#                z = z + t_active
#                next_event = [z, ID, k, 4]
#                t_forthcoming.add(next_event)
#                # 2.- Desde las otras CETIs hacia la recién agregada
#                t_hola = CETIs[2][3]
#                t = min(z - t_hola, t_chau)   #### PENSAR BIEN ESTO
#                next_event = [t, ID, k, 4]
#                t_forthcoming.add(next_event)

        # agregar la CETI a la lista de CETIs activas
        # [ID, x, y, t_hola, t_chau]
        #print('agregando a CHATs, en el indice: %d' % len(CHATs) )
        CHATs.append([x, y])
        CHATs_idx.append(ID)
        print('APPENDING..... %d %d' % (len(CHATs), ID))


        # AGREGAR A CETIs

        # rehacer el árbol
        tree = sp.cKDTree( data=CHATs ) 

        # eliminar el tiempo actual
        t_forthcoming.remove_first()

        #}}}
            
    if case==2:
        #{{{
        #-------------------------------------
        # END CETI
        #-------------------------------------

        # eliminar la CETI a la lista de CETIs activas
        # [ID, x, y, t_hola, t_chau]
        try:
            id_loc = CHATs_idx.index(ID)
            del CHATs[id_loc]
            del CHATs_idx[id_loc]
        except TypeError:
            pass

        # rehacer el árbol
        if len(CHATs)>0:
            tree = sp.cKDTree( data=CHATs ) 

        # eliminar el tiempo actual
        t_forthcoming.remove_first()
        
        #}}}
        
    if case==3:
        #{{{
        #-------------------------------------
        # NEW CONTACT
        #-------------------------------------
        
        # eliminar la CETI a la lista de CETIs activas
        # [ID, x, y, t_hola, t_chau]
        ID = 1
        t_chau = t_now

        CETIs[ID] = [(x, y, t_hola, t_chau)]

        # rehacer el árbol
        M = np.column_stack([x, y])
        tree = sp.cKDTree( data=M ) 

        # eliminar el tiempo actual
        t_forthcoming.remove_first()
        #}}}
        
    #if case==4:
        #{{{
        #-------------------------------------
        # END CONTACT
        #-------------------------------------

        # eliminar la CETI a la lista de CETIs activas
        # [ID, x, y, t_hola, t_chau]
        ID = 1
        t_chau = t_now

        CETIs[ID] = [(x, y, t_hola, t_chau)]

        # rehacer el árbol
        M = np.column_stack([x, y])
        tree = sp.cKDTree( data=M ) 

        # eliminar el tiempo actual
        t_forthcoming.remove_first() 

        #}}}

    t_forthcoming.show()
    #time.sleep(2)
    raw_input("\n...")
        
#}}}









#print('Time: %f | ID: %d | case: %d' % (t_now, ID, case))

#print ('Size of CHATs: %d  | ID: %d' % (len(CHATs), ID))

#print('t_next_awake.: %f' % t_next_awakening)
