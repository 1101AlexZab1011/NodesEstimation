class Subject(object):
    def __init__(self, name, data, nodes, directory, dataset):
        self.name = name
        if not isinstance(data, dict):
            raise ValueError('Subject\'s data must be a dictionary with data types as keys and files paths as values')
        self.data = data
        self.nodes = nodes
        self.dir = directory
        self.dataset = dataset

    def set_data(self, data):
        self.data.update(data)
