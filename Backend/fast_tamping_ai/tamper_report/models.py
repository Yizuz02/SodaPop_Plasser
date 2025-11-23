from django.db import models
from django.contrib.auth.models import User

class Station(models.Model):
    name = models.CharField(max_length=100)
    city = models.CharField(max_length=100)
    latitude = models.FloatField()
    longitude = models.FloatField()

    def __str__(self):
        return f"{self.name} - {self.city}"

class Train(models.Model):
    TRAIN_TYPE = (
        ('passenger', 'Passenger'),
        ('cargo', 'Cargo'),
        ('highspeed', 'High-Speed'),
    )
    model = models.CharField(max_length=100)  
    weight = models.FloatField()              
    max_speed = models.FloatField()           

    train_type = models.CharField(max_length=20, choices=TRAIN_TYPE)
    manufacturer = models.CharField(max_length=100)
    year_built = models.IntegerField()

    def __str__(self):
        return f"{self.id} ({self.model}) - {self.get_train_type_display()}"

class Route(models.Model):
    route_code = models.CharField(max_length=20, unique=True)
    origin = models.ForeignKey(
        Station, related_name="route_origin", on_delete=models.CASCADE
    )
    destination = models.ForeignKey(
        Station, related_name="route_destination", on_delete=models.CASCADE
    )

    def __str__(self):
        return f"{self.route_code}: {self.origin.name} → {self.destination.name}"

class TrainTrip(models.Model):
    train = models.ForeignKey(Train, on_delete=models.CASCADE)
    route = models.ForeignKey(Route, on_delete=models.CASCADE)

    cargo_weight = models.FloatField()    
    average_speed = models.FloatField()    

    departure_time = models.DateTimeField()
    arrival_time = models.DateTimeField()

    def __str__(self):
        return f"{self.train.id} Trip on {self.route.route_code} ({self.departure_time.date()})"

class ReportBatch(models.Model):
    train_trip = models.OneToOneField(TrainTrip, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Batch for Trip {self.train_trip.id} ({self.train_trip.route.route_code})"

class Report(models.Model):
    STATE_CHOICES = (
        (1, 'Pending'),
        (2, 'In Process'),
        (3, 'Fixed'),
        (4, 'Unsolved'),
    )

    report_batch = models.ForeignKey(ReportBatch, on_delete=models.CASCADE)
    
    start_latitude = models.FloatField()
    start_longitude = models.FloatField()

    end_latitude = models.FloatField()
    end_longitude = models.FloatField()

    lift_left_mm = models.FloatField()
    lift_right_mm = models.FloatField()
    adjustment_left_mm = models.FloatField()
    adjustment_right_mm = models.FloatField()

    date_reported = models.DateTimeField(auto_now_add=True)
    date_fixed = models.DateTimeField(null=True, blank=True)

    state = models.IntegerField(choices=STATE_CHOICES, default=1)

    def __str__(self):
        return f"Report on route {self.report_batch.train_trip.route} | Lift: {self.lift_mm} mm"
    
class TamperMachine(models.Model):

    TAMPER_TYPE = (
    # Bateadoras de Vía y Desvíos (equivalente a 'unimat')
    ('unimat_4s', 'Unimat 4S (Universal)'),
    ('unimat_08_475', 'Unimat 08-475/4S (Desvíos)'),
    
    # Bateadoras de Línea de Alto Rendimiento (equivalente a '09-3x' o 'plain_line')
    ('09_3x_dyn', '09-3X Dynamic (Alto Rendimiento)'),
    ('09_32_csm', '09-32 CSM (Línea Continua)'),
    
    # Bateadoras Dynamic (Estabilización y Tampeo - equivalente a 'dynamic')
    ('dynamic_9000', 'Dynamic 9000 (Estabilización/Tampeo)'),
    ('pms_2030', 'PMS 2030 (Plasser Measuring System)'),
    
    # Bateadoras de Línea Estándar (equivalente a 'plain_line')
    ('duomatic_09', 'Duomatic 09 (Doble Bateadora)'),
)
    model = models.CharField(max_length=100)
    manufacturer = models.CharField(max_length=100)
    year_built = models.IntegerField()

    weight = models.FloatField()
    max_speed = models.FloatField()          
    working_speed = models.FloatField()      

    tamper_type = models.CharField(max_length=20, choices=TAMPER_TYPE)

    def __str__(self):
        return f"{self.id} ({self.model}) - {self.get_tamper_type_display()}"
    
class TamperOperation(models.Model):
    tamper = models.ForeignKey(TamperMachine, on_delete=models.CASCADE)
    report_batch = models.ForeignKey(ReportBatch, on_delete=models.CASCADE)

    start_time = models.DateTimeField()
    end_time = models.DateTimeField(blank=True, null=True)

    operator = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
    )

    def __str__(self):
        return (
            f"{self.tamper.id} Operation on {self.report_batch.train_trip.route.route_code} "
            f"({self.start_time.date()})"
        )

