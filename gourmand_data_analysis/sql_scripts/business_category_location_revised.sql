with business_locations as ( 
    select 
    db.BusinessKey,
    db.BusinessName,
    db.ChainName,
    db.PaymentLevelName,
    db.Longitude,
    db.Latitude,
    dl.CityName,
    dl.CountyName,
    dl.StateName,
    dl.CountryName
from `gourmanddwh.g_production.DimBusiness` db 
join `gourmanddwh.g_production.DimLocation` dl on db.CitySourceKey=dl.CitySourceKey
),
business_cats as (
select 
    db.BusinessKey,
    db.BusinessName,
    string_agg(dbc.BusinessCategoryName, ', ') business_cat_list,
    count(dbc.BusinessCategoryName) cat_counts
from `gourmanddwh.g_production.DimBusinessCategoryBridge` bcb
join `gourmanddwh.g_production.DimBusiness` db on bcb.BusinessKey = db.BusinessKey
join `gourmanddwh.g_production.DimBusinessCategory` dbc on bcb.businesscategorykey = dbc.businesscategorykey
group by db.BusinessKey, db.BusinessName
)
select 
    bl.*,
    bc.business_cat_list,
    bc.cat_counts
from business_locations bl
left join business_cats bc on bl.BusinessKey = bc.BusinessKey