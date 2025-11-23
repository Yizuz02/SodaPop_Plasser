from django.contrib.auth.models import User
from .models import Station, Train, TrainTrip, TamperMachine, TamperOperation, Report, ReportBatch, Route 
from rest_framework import viewsets, permissions
from .serializers import UserSerializer, StationSerializer, TrainSerializer, TamperOperationSerializer, TamperMachineSerializer, TrainTripSerializer, ReportBatchSerializer, ReportSerializer, RouteSerializer

class UserViewSet(viewsets.ModelViewSet):
    queryset = User.objects.all()
    serializer_class = UserSerializer
    permission_classes = [permissions.IsAdminUser]

class StationViewSet(viewsets.ModelViewSet):
    queryset = Station.objects.all()
    serializer_class = StationSerializer
    def get_permissions(self):
        if self.action in ['create', 'update', 'partial_update', 'destroy']:
            return [permissions.IsAdminUser()]
        
        return [permissions.IsAuthenticated()]
    
class TrainViewSet(viewsets.ModelViewSet):
    queryset = Train.objects.all()
    serializer_class = TrainSerializer
    def get_permissions(self):
        if self.action in ['create', 'update', 'partial_update', 'destroy']:
            return [permissions.IsAdminUser()]
        
        return [permissions.IsAuthenticated()]

    
class TrainTripViewSet(viewsets.ModelViewSet):
    queryset = TrainTrip.objects.all()
    serializer_class = TrainTripSerializer
    def get_permissions(self):
        if self.action in ['create', 'update', 'partial_update', 'destroy']:
            return [permissions.IsAdminUser()]
        
        return [permissions.IsAuthenticated()]
    
class RouteViewSet(viewsets.ModelViewSet):
    queryset = Route.objects.all()
    serializer_class = RouteSerializer
    def get_permissions(self):
        if self.action in ['create', 'update', 'partial_update', 'destroy']:
            return [permissions.IsAdminUser()]
        
        return [permissions.IsAuthenticated()]
    
    
class ReportViewSet(viewsets.ModelViewSet):
    queryset = Report.objects.all()
    serializer_class = ReportSerializer
    def get_permissions(self):
        if self.action in ['create', 'update', 'partial_update', 'destroy']:
            return [permissions.IsAdminUser()]
        
        return [permissions.IsAuthenticated()]