"""OLISTS.

Module for the manipulation of ordered lists.
This module is internal to hearsay.
"""


class Node:
    """Node and linked list classes.

    This class contains tools to manipulate nodes.  A node is
    a point in the Galaxy that a acquires the ability to emit
    and receive messages at a given time.  A set of nodes make
    a linked list.
    """

    def __init__(self, data):
        """Initialize a node.

        Parameters
        ----------
            data: (single value)
            A number or value that can be compared and supports
            the <grater than> operator.
        Returns
        -------
            None
        Raises
        ------
            None
        """
        self.data = data
        self.next = None

    def getData(self):
        """Get data in a node.

        """
        return self.data

    def getNext(self):
        """Get the next node, if exists.

        """
        return self.next

    def setNext(self, newnext):
        """Set the next node.

        """
        self.next = newnext


class OrderedList:
    """Ordered list class.

    Tools to make ordered lists. This structure is useful because it can be
    traversed and a new node can be added at any stage.
    # based on http://interactivepython.org/courselib/static/pythonds/
    #  BasicDS/ImplementinganOrderedList.html
    """

    def __init__(self):
        """Initialize ordered list.

        """
        self.head = None
        self.last = None

    def show(self):
        """Print an ordered list.

        """
        state = ['awakening', 'doomsday', 'contact ', 'blackout']
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

            current = current.getNext()

    def add(self, data):
        """Add an element to an ordered list.

        """
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
        """Remove first element.

        """
        self.head = self.head.getNext()

    def isEmpty(self):
        """Ask if list is empty.

        """
        return self.head is None

    def size(self):
        """Retrieve the size of the list.

        """
        current = self.head
        count = 0
        while current is not None:
            count = count + 1
            current = current.getNext()
        return count
