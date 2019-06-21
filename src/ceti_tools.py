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
        state = ['awakening','doomsday','contact ','linedown']
        current = self.head
        while current is not None:

            if current.getData()[2] is None:
                print("%7.2f -- case %i (%8r) -- id: %i " %
                        tuple([current.getData()[0],
                            current.getData()[3], 
                            state[current.getData()[3]-1], 
                            current.getData()[1]]))
            else:
                print("%7.2f -- case %i -- %i -> %i" % tuple([current.getData()[0], current.getData()[3], current.getData()[1], current.getData()[2]]))

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
    print('\n\n')
    for i in range(len(CETIs)):
        print('%3d          (%5.0f, %5.0f) lyr  <%5.0f, %5.0f> yr' %
              CETIs[i][0][1:])
        k = len(CETIs[i]) - 1
        for l in range(k):
            print('%3d sees %3d (%5.0f, %5.0f) lyr  <%5.0f, %5.0f> yr' %
                  CETIs[i][l+1])

            Dx = np.sqrt(((
                np.array(CETIs[i][0][2:4]) - 
                np.array(CETIs[i][l+1][2:4]))**2).sum())

# }}}

# from importlib import reload
# ceti_tools = reload(ceti_tools)
