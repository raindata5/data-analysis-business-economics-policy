select
    dl.CountyName,
    fcg.EstimationYear,
    fcg.EstimatedPopulation
from `gourmanddwh.g_production.FactCountyGrowth` fcg
join `gourmanddwh.g_production.DimLocation` dl on fcg.CountySourceKey=dl.CountySourceKey
where fcg.EstimationYear=2019;