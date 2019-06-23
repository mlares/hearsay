import numpy as np

# Resources:
# http://interactivepython.org/courselib/static/pythonds/BasicDS/ImplementinganOrderedList.html


# NODE AND LINKED LIST CLASSES
# {{{
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

    def getData(self):
        return self.data

    def getNext(self):
        return self.next

    def setNext(self, newnext):
        self.next = newnext


class OrderedList:
    def __init__(self):
        self.head = None
        self.last = None

    def show(self):
        state = ['awakening','doomsday','contact ','blackout']
        current = self.head
        while current is not None:

            if current.getData()[2] is None:
                print("   %6.2f  %14s - %i " %
                        tuple([current.getData()[0],
                            state[current.getData()[3]-1], 
                            current.getData()[1]]))
            else:
                print("   %6.2f  %14s - %i <    %i" % 
                        tuple([current.getData()[0], 
                            state[current.getData()[3]-1], 
                            current.getData()[1], 
                            current.getData()[2]]))

#            print(current.getData())


            current = current.getNext()        

    def add(self, data):
        current = self.head
        previous = None
        stop = False
        while current is not None and not stop:
            if current.getData()[0] > data[0]:
                stop = True
            else:
                previous = current
                current = current.getNext() 
        temp = Node(data)
        if previous is None:
            temp.setNext(self.head)
            self.head = temp
        else:
            temp.setNext(current)
            previous.setNext(temp)

    def remove_first(self):
        self.head = self.head.getNext()

    def isEmpty(self):
        return self.head is None

    def size(self):
        current = self.head
        count = 0
        while current is not None:
            count = count + 1
            current = current.getNext()
        return count
# }}}

# PERSONALIZED FUNCTIONS
# {{{

def ShowCETIs(CETIs):
    for i in range(len(CETIs)):
        print('%2d         (%5.0f, %5.0f) yr      <%5.0f, %5.0f> lyr' %
                 (CETIs[i][0][1], CETIs[i][0][4],
                  CETIs[i][0][5], CETIs[i][0][2], CETIs[i][0][3]))

        k = len(CETIs[i]) - 1
        for l in range(k):
            Dx = np.sqrt(((
                np.array(CETIs[i][0][2:4]) - 
                np.array(CETIs[i][l+1][2:4]))**2).sum())

            print('%2d sees %2d (%5.0f, %5.0f) yr      <%5.0f, %5.0f> lyr distance=%f' % (CETIs[i][l+1][0], CETIs[i][l+1][1], CETIs[i][l+1][4], CETIs[i][l+1][5], CETIs[i][l+1][2], CETIs[i][l+1][3], Dx))

# }}}

# from importlib import reload
# ceti_tools = reload(ceti_tools)
