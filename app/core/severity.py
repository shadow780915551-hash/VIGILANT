
class SeverityCalculator:
    @staticmethod
    def calculate(confidence, num_detections, time_in_zone=0):
        score = confidence * 100
        if num_detections > 1:
            score += (num_detections - 1) * 20
        if time_in_zone > 5:
            score += (time_in_zone - 5) * 10
        
        if score < 60:
            return "LOW"
        elif score < 85:
            return "MEDIUM"
        else:
            return "HIGH"
