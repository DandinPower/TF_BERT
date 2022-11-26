import csv

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
        print(f'shifts: {self.shifts}, latency: {self.shifts * 5e-10}, energy: {self.shifts * 20}; detects: {self.detects}, latency: {self.detects * 1e-10}, energy: {self.detects * 2}; inserts: {self.inserts}, latency: {self.inserts * 1e-9}, energy: {self.inserts * 200}; removes: {self.removes}, latency: {self.removes * 8e-10}, energy: {self.removes * 20}; total latency: {self.CountLatency()}s, total energy: {self.CountEnergy()}fJ')

    def WriteShow(self, path):
        with open(path, 'w') as csvfile:
            fieldnames = ['shifts', 'shifts_latency', 'shifts_energy', 'detects', 'detects_latency', 'detects_energy', 'inserts', 'inserts_latency', 'inserts_energy', 'removes', 'removes_latency', 'removes_energy', 'total_latency', 'total_energy']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow({'shifts': f'{self.shifts}', 'shifts_latency': f'{self.shifts * 5e-10}', 'shifts_energy': f'{self.shifts * 20}', 'detects': f'{self.detects}', 'detects_latency': f'{self.detects * 1e-10}', 'detects_energy': f'{self.detects * 2}', 'inserts': f'{self.inserts}', 'inserts_latency': f'{self.inserts * 1e-9}', 'inserts_energy': f'{self.inserts * 200}', 'removes': f'{self.removes}', 'removes_latency': f'{self.removes * 8e-10}', 'removes_energy': f'{self.removes * 20}', 'total_latency': f'{self.CountLatency()}', 'total_energy': f'{self.CountEnergy()}'})