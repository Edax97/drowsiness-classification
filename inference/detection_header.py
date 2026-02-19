SOMNOLENCIA_DET = "somnolencia"
ALERTA_DET = "alerta"
INDEFINIDO_DET = ""
class Detection_Header:
    def __init__(self, drowsy_class: str, awake_class: str, score=0.0) -> None:
        self.detection_history = ["", "", ""]
        self.drowsy_class = drowsy_class
        self.awake_class = awake_class
        self.status = INDEFINIDO_DET
        self.score = score

    def set_status(self, class_name: str, score: float=1) -> str:
        self.detection_history[:-1]= self.detection_history[1:]
        self.detection_history[-1] = class_name

        drowsy_det_num = 0
        for det in self.detection_history:
            if det == self.drowsy_class:
                drowsy_det_num += 1
        if drowsy_det_num > 1:
            self.status = SOMNOLENCIA_DET
            return self.status
        if class_name == self.awake_class and score > self.score:
            self.status = ALERTA_DET
            return self.status
        self.status = INDEFINIDO_DET
        return self.status

    def get_status(self):
        return self.status



