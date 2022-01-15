with each_county as (
select distinct
    dl.CountyName,
    fcg.EstimationYear,
    fcg.EstimatedPopulation,
    fcg.CountySourceKey,
    dl.StateName
from `gourmanddwh.g_production.FactCountyGrowth` fcg
join `gourmanddwh.g_production.DimCounty` dl on fcg.CountySourceKey=dl.CountySourceKey
where fcg.EstimationYear in (2018,2019)
),
last_pops as (
select 
    ec.CountyName,
    ec.EstimationYear,
    ec.EstimatedPopulation,
    ec.StateName,
    LAG(ec.EstimatedPopulation) OVER (partition by ec.StateName, ec.CountyName order by ec.EstimationYear ASC) lastpop
from each_county ec 
)
select 
    lp.StateName,
    lp.CountyName,
    lp.EstimationYear,
    lp.EstimatedPopulation,
    lp.lastpop,
    (lp.EstimatedPopulation - lp.lastpop) / lp.lastpop relative_delta,
    (lp.EstimatedPopulation - lp.lastpop) abs_delta
from last_pops lp
where lp.EstimationYear = 2019