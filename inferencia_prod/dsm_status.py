import time

class DSMstatus:

    def __init__(self, send_interval: float, minimum_interval: float):
        self.send_interval = send_interval
        self.minimum_interval = minimum_interval
        
        self.last_telefono = False
        self.last_cigarro = False
        self.last_cinturon = False
        self.last_gafas = False
        self.last_fatiga = False
        self.last_distraccion = False        

        self.telefono = False
        self.cigarro = False
        self.cinturon = False
        self.gafas = False
        self.fatiga = False
        self.distraccion = False
        
        self.driver = False
        self.time_sent = time.time()

    def _set_last_status(self):
        self.last_cigarro = self.cigarro
        self.last_telefono = self.telefono
        self.last_cinturon = self.cinturon
        self.last_gafas = self.gafas
        self.last_distraccion = self.distraccion
        self.last_fatiga = self.fatiga

    def set_status(
        self,
        driver: bool,
        telefono = False,
        cigarro = False,
        cinturon = True,
        gafas = True,
        distraccion = False,
        fatiga = False
    ):
        self._set_last_status()

        self.telefono = telefono
        self.cigarro = cigarro
        self.cinturon = cinturon
        self.gafas = gafas
        self.distraccion = distraccion
        self.fatiga = fatiga
        self.driver = driver

    def get_status_string(self) -> str:
        if not self.driver:
            return ""

        current_time = time.time()
        if current_time - self.time_sent < self.minimum_interval:
            return ""

        status_string = f"DSM,TEL:{int(self.telefono)},CIG:{int(self.cigarro)},CINTURON:{int(self.cinturon)},GAFAS:{int(self.gafas)},DIS:{int(self.distraccion)},FATIGA:{int(self.fatiga)}"
        if current_time - self.time_sent > self.send_interval:
            self.time_sent = current_time
            self._set_last_status()
            return status_string 

        if  (self.telefono and not self.last_telefono) or \
            (self.cigarro and not self.last_cigarro) or \
            (not self.cinturon and self.last_cinturon) or \
            (not self.gafas and self.last_gafas) or \
            (self.distraccion and not self.last_distraccion) or \
            (self.fatiga and not self.last_fatiga):
            self.time_sent = current_time
            self._set_last_status()
            return status_string

        return ""
            
