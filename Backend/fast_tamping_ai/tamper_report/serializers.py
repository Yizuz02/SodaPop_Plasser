from django.contrib.auth.models import User
from .models import Report, ReportBatch, Station, TamperMachine, TamperOperation, Train, Route, TrainTrip
from rest_framework import serializers

class UserSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True)

    class Meta:
        model = User
        fields = ['id', 'username', 'password', 'first_name', 'last_name']

    def create(self, validated_data):
        password = validated_data.pop('password')
        user = User(**validated_data)
        user.set_password(password)
        user.save()
        return user

    def update(self, instance, validated_data):
        password = validated_data.pop('password', None)
        for attr, value in validated_data.items():
            setattr(instance, attr, value)
        if password:
            instance.set_password(password)
        instance.save()
        return instance
    

class StationSerializer(serializers.ModelSerializer):
    class Meta:
        model = Station
        fields = ['id', 'name', 'city', 'latitude', 'longitude']


class TrainSerializer(serializers.ModelSerializer):
    class Meta:
        model = Train
        fields = ['id','model','weight','max_speed','train_type','manufacturer','year_built']

class RouteSerializer(serializers.ModelSerializer):
    origin = StationSerializer(read_only=True)
    destination = StationSerializer(read_only=True)

    class Meta:
        model = Route
        fields = ['id', 'route_code', 'origin', 'destination']


class TrainTripSerializer(serializers.ModelSerializer):
    train = TrainSerializer(read_only=True)
    route = RouteSerializer(read_only=True)

    class Meta:
        model = TrainTrip
        fields = [
            'id', 'train', 'route',
            'cargo_weight', 'average_speed',
            'departure_time', 'arrival_time'
        ]

class ReportBatchSerializer(serializers.ModelSerializer):
    train_trip = TrainTripSerializer(read_only=True)

    class Meta:
        model = ReportBatch
        fields = ['id', 'train_trip', 'created_at']

class ReportSerializer(serializers.ModelSerializer):
    report_batch = serializers.PrimaryKeyRelatedField(
        queryset=ReportBatch.objects.all()
    )

    class Meta:
        model = Report
        fields = [
            'id', 'report_batch',
            'start_latitude', 'start_longitude',
            'end_latitude', 'end_longitude',
            'lift_left_mm', 'lift_right_mm',
            'date_reported', 'date_fixed',
            'state'
        ]

class TamperMachineSerializer(serializers.ModelSerializer):
    class Meta:
        model = TamperMachine
        fields = '__all__'

class TamperOperationSerializer(serializers.ModelSerializer):
    tamper = TamperMachineSerializer(read_only=True)
    report_batch = ReportBatchSerializer(read_only=True)
    operator = UserSerializer(read_only=True)

    class Meta:
        model = TamperOperation
        fields = [
            'id', 'tamper', 'report_batch',
            'start_time', 'end_time', 'operator'
        ]



