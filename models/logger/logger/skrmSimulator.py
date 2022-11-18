class SKRM:
    def __init__(self):
        self.shifts = 0
        self.detects = 0
        self.inserts = 0
        self.removes = 0

    def Add(self, _record):
        self.shifts += _record[0]
        self.detects += _record[1]
        self.inserts += _record[2]
        self.removes += _record[3]
    
    def Show(self):
        print(f'shifts: {self.shifts}, detects: {self.detects}, inserts: {self.inserts}, removes: {self.removes}')