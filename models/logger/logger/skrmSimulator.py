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

    def CountLatency(self):
        shiftsLatency = 5e-10
        detectsLatency = 1e-10
        removesLatency = 8e-10
        insertsLatency = 1e-9
        return self.shifts * shiftsLatency + self.detects * detectsLatency + self.removes * removesLatency + self.inserts * insertsLatency
 
    def CountEnergy(self):
        shiftsEnergy = 20
        detectsEnergy = 2
        removesEnergy = 20
        insertsEnergy = 200
        return self.shifts * shiftsEnergy + self.detects * detectsEnergy + self.removes * removesEnergy + self.inserts * insertsEnergy

    def Show(self):
        print(f'shifts: {self.shifts}, detects: {self.detects}, inserts: {self.inserts}, removes: {self.removes}, latency: {self.CountLatency()}s, energy: {self.CountEnergy()}fJ')