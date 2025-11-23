from django.db import models


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

    name = models.CharField(max_length=100)
    model = models.CharField(max_length=100)  
    weight = models.FloatField()              
    max_speed = models.FloatField()           

    train_type = models.CharField(max_length=20, choices=TRAIN_TYPE)
    manufacturer = models.CharField(max_length=100)
    year_built = models.IntegerField()

    def __str__(self):
        return f"{self.name} ({self.model}) - {self.get_train_type_display()}"

class Route(models.Model):
    route_code = models.CharField(max_length=20, unique=True)
    origin = models.ForeignKey(
        Station, related_name="route_origin", on_delete=models.CASCADE
    )
    destination = models.ForeignKey(
        Station, related_name="route_destination", on_delete=models.CASCADE
    )

    def __str__(self):
        return f"{self.route_code}: {self.origin.code} → {self.destination.code}"

class TrainTrip(models.Model):
    train = models.ForeignKey(Train, on_delete=models.CASCADE)
    route = models.ForeignKey(Route, on_delete=models.CASCADE)

    cargo_weight = models.FloatField()    
    average_speed = models.FloatField()    

    departure_time = models.DateTimeField()
    arrival_time = models.DateTimeField()


    def __str__(self):
        return f"{self.train.name} Trip on {self.route.route_code} ({self.departure_time.date()})"

class Report(models.Model):
    STATE_CHOICES = (
        (1, 'Pending'),
        (2, 'In Process'),
        (3, 'Fixed'),
        (4, 'Unsolved'),
    )

    train_trip = models.ForeignKey(TrainTrip, on_delete=models.CASCADE)
    
    start_latitude = models.FloatField()
    start_longitude = models.FloatField()

    end_latitude = models.FloatField()
    end_longitude = models.FloatField()

    lift_mm = models.FloatField()

    date_reported = models.DateTimeField(auto_now_add=True)
    date_fixed = models.DateTimeField(null=True, blank=True)

    state = models.IntegerField(choices=STATE_CHOICES, default=1)

    def __str__(self):
        return f"Report on route {self.train_trip.route.route_code} | Lift: {self.lift_mm} mm"