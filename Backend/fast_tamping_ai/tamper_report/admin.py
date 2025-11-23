from django.contrib import admin
from .models import Station, Train, Route, TrainTrip, Report


@admin.register(Station)
class StationAdmin(admin.ModelAdmin):
    list_display = ("name", "city", "latitude", "longitude")
    search_fields = ("name", "city")


@admin.register(Train)
class TrainAdmin(admin.ModelAdmin):
    list_display = ("name", "model", "train_type", "manufacturer", "year_built", "max_speed", "weight")
    search_fields = ("name", "model", "manufacturer")
    list_filter = ("train_type", "manufacturer", "year_built")


@admin.register(Route)
class RouteAdmin(admin.ModelAdmin):
    list_display = ("route_code", "origin", "destination")
    search_fields = ("route_code",)
    list_filter = ("origin", "destination")


@admin.register(TrainTrip)
class TrainTripAdmin(admin.ModelAdmin):
    list_display = (
        "train",
        "route",
        "cargo_weight",
        "average_speed",
        "departure_time",
        "arrival_time",
    )
    list_filter = ("train", "route")
    search_fields = ("train__name", "route__route_code")


@admin.register(Report)
class ReportAdmin(admin.ModelAdmin):
    list_display = (
        "train_trip",
        "lift_mm",
        "state",
        "date_reported",
        "date_fixed",
        "start_latitude",
        "start_longitude",
        "end_latitude",
        "end_longitude",
    )
    list_filter = ("state", "train_trip__route", "date_reported")
    search_fields = ("train_trip__train__name", "train_trip__route__route_code")
