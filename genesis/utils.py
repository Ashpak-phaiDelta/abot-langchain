
def read_input():
    '''Simple prompt for user to enter input for asking'''
    while True:
        inpt = input('> ')
        if len(inpt) > 0:
            return inpt
