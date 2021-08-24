import src.models.util.point_lib as pointmodels


def Point(model, u_obs, u):
    return getattr(pointmodels, model)(u_obs, u)