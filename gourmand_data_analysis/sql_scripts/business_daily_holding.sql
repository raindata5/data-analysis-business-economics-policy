with bus_lag as (
select 
    db.BusinessName,
    db.ChainName,
    fbh.BusinessRating,
    fbh.ReviewCount,
    fbh.CloseDate,
    lag(fbh.ReviewCount) OVER (partition by db.BusinessName order by fbh.CloseDate) previous_review_cnt,
    lag(fbh.BusinessRating) OVER (partition by db.BusinessName order by fbh.CloseDate) previous_rating
from `gourmanddwh.g_production.FactBusinessHolding` fbh
join `gourmanddwh.g_production.DimBusiness` db on fbh.businesskey=db.BusinessKey
where db.is_current = 1
order by db.BusinessName, fbh.CloseDate
)
,
total_lags as (
SELECT
    BusinessName,
    ChainName,
    BusinessRating,
    ReviewCount,
    CloseDate,
    previous_review_cnt,
    previous_rating,
    ReviewCount - previous_review_cnt abs_review_diff,
    BusinessRating - previous_rating abs_rating_diff,
    SUM(ReviewCount - previous_review_cnt) OVER(partition by BusinessName) total_review_cnt_delta,
    SUM(BusinessRating - previous_rating) OVER(partition by BusinessName) total_bus_rating_delta
from bus_lag 
order by BusinessName, CloseDate
)
select 
    *
from total_lags