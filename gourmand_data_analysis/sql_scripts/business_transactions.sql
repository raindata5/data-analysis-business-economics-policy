-- select 
--     b.BusinessKey,
--     b.BusinessName,
--     count(dtt.TransactionName) over (partition by b.BusinessKey) transactiontype,
--     dtt.TransactionName
-- from `gourmanddwh.g_production.DimBusiness` b
-- left join `gourmanddwh.g_production.DimBusinessTransactionBridge` btb on b.BusinessKey = btb.BusinessKey 
-- left join `gourmanddwh.g_production.DimTransactionType` dtt on btb.TransactionKey = dtt.TransactionKey;

select 
    b.BusinessKey,
    b.BusinessName,
    count(dtt.TransactionName) transactioncounts,
    coalesce(string_agg(dtt.TransactionName, ', '), 'other') transactions_list
from `gourmanddwh.g_production.DimBusiness` b
left join `gourmanddwh.g_production.DimBusinessTransactionBridge` btb on b.BusinessKey = btb.BusinessKey 
left join `gourmanddwh.g_production.DimTransactionType` dtt on btb.TransactionKey = dtt.TransactionKey
group by b.BusinessKey, b.BusinessName ;