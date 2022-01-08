select 
    db.BusinessName,
    db.ChainName,
    fbh.BusinessRating,
    fbh.ReviewCount
from `gourmanddwh.g_production.FactBusinessHolding` fbh
join `gourmanddwh.g_production.DimBusiness` db on fbh.businesskey=db.BusinessKey
where db.is_current = 1;