from rest_framework.routers import DefaultRouter
from .views import UserViewSet, StationViewSet, TrainViewSet

router = DefaultRouter()
router.register(r'users', UserViewSet)
router.register(r'stations', StationViewSet)
router.register(r'trains', TrainViewSet)

urlpatterns = router.urls
