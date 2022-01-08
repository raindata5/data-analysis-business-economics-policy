select 
    db.BusinessKey,
    db.BusinessName,
    db.ChainName,
    db.PaymentLevelName,
    db.Longitude,
    db.Latitude,
    dbc.BusinessCategoryName,
    dl.CityName,
    dl.CountyName,
    dl.StateName,
    dl.CountryName
from `gourmanddwh.g_production.DimBusinessCategoryBridge` bcb
join `gourmanddwh.g_production.DimBusiness` db on bcb.BusinessKey = db.BusinessKey
join `gourmanddwh.g_production.DimBusinessCategory` dbc on bcb.businesscategorykey = dbc.businesscategorykey
join `gourmanddwh.g_production.DimLocation` dl on db.CitySourceKey=dl.CitySourceKey;