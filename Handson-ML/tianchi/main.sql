--------------------------------------------------------------------------------
--                              参数说明                                     --
--------------------------------------------------------------------------------
--集成的数据处理流程，包含了特征提取，数据融合，加分类标签，归一化和欠采样等
--输入参数：
--label_day：分类标签的日期，如：'2014-12-18 00'
--table_label:保存的表名label，例如16
--输出表：
--特征表系列：${table_label}_item_features_1,...,${table_label}_item_features_n,${table_label}_item_features,
--           ${table_label}_ui_features_1,....,${table_label}_ui_features_n,${table_label}_ui_features,
--           ${table_label}_user_features_1,....,${table_label}_user_features_n,${table_label}_user_features,
 -- sum(case when(behavior_type=1 andtime<'${label_day}') then 1 else 0 end) as i1,
 --数据融合处理：
--           ${table_label}_feature_table,${table_label}_normal,${table_label}_under_sample
 
 
--------------------------------------------------------------------------------
--                              特征提取                                     --
--------------------------------------------------------------------------------
 
--------------------------------------------------------------------------------
--使用label_day之前的数据生成item特征
 
--已成功
drop table if exists${table_label}_item_features_1;
create table${table_label}_item_features_1 as
select
    item_id,
    --1）对不同item点击、收藏、购物车、购买的总计
    sum(case when(behavior_type=2 and time<'${label_day}') then 1 else 0 end) as i2,
    sum(case when(behavior_type=3 and time<'${label_day}') then 1 else 0 end) as i3,
    sum(case when(behavior_type=4 and time<'${label_day}') then 1 else 0 end) as i4,
    --2)对不同item点击、收藏、购物车、购买平均每个user的计数
    sum(case when(behavior_type=1 and time<'${label_day}') then 1 else 0 end)/count(distinct user_id) as i5,
    --2015-5-6添加（品牌是否有变热门的征兆）
    --最近第1天的行为数与日平均行为数的比值
    case when sum(case when behavior_type=1 then 1 else 0 end)=0 then 0 else 
        -- 最近一天行为数 / (行为总数/天数)
        sum(case when behavior_type=1 and 
            datediff(to_date('${label_day}','yyyy-mm-dd hh'),to_date(time,'yyyy-mm-ddhh'),'dd')=1 then 1 else 0 end
        )
        -- 天数
        * datediff(to_date('${label_day}','yyyy-mm-ddhh'),to_date('2014-11-18 00','yyyy-mm-dd hh'),'dd') /
        -- 行为总数
        sum(case when behavior_type=1 then 1 else 0 end) end as i6,
    case when sum(case when behavior_type=2 then 1 else 0 end)=0 then 0 else sum(case when behavior_type=2 and datediff(to_date('${label_day}','yyyy-mm-dd hh'),to_date(time,'yyyy-mm-ddhh'),'dd')=1 then 1 else 0 end)*datediff(to_date('${label_day}','yyyy-mm-ddhh'),to_date('2014-11-18 00','yyyy-mm-dd hh'),'dd')/sum(case when behavior_type=2 then 1 else 0 end) end as i7,
    case when sum(case when behavior_type=3 then 1 else 0 end)=0 then 0 else sum(case when behavior_type=3 and datediff(to_date('${label_day}','yyyy-mm-dd hh'),to_date(time,'yyyy-mm-ddhh'),'dd')=1 then 1 else 0 end)*datediff(to_date('${label_day}','yyyy-mm-ddhh'),to_date('2014-11-18 00','yyyy-mm-dd hh'),'dd')/sum(case when behavior_type=3 then 1 else 0 end) end as i8,
    --最近第2天的行为数与日平均行为数的比值
    case when sum(case when behavior_type=1then 1 else 0 end)=0 then 0 else sum(case when behavior_type=1 and datediff(to_date('${label_day}','yyyy-mm-dd hh'),to_date(time,'yyyy-mm-ddhh'),'dd')=2 then 1 else 0 end)*datediff(to_date('${label_day}','yyyy-mm-ddhh'),to_date('2014-11-18 00','yyyy-mm-dd hh'),'dd')/sum(case when behavior_type=1 then 1 else 0 end) end as i9,
    case when sum(case when behavior_type=2then 1 else 0 end)=0 then 0 else sum(case when behavior_type=2 and datediff(to_date('${label_day}','yyyy-mm-dd hh'),to_date(time,'yyyy-mm-ddhh'),'dd')=2 then 1 else 0 end)*datediff(to_date('${label_day}','yyyy-mm-ddhh'),to_date('2014-11-18 00','yyyy-mm-dd hh'),'dd')/sum(case whenbehavior_type=2 then 1 else 0 end) end as i10,
    case when sum(case when behavior_type=3then 1 else 0 end)=0 then 0 else sum(case when behavior_type=3 and datediff(to_date('${label_day}','yyyy-mm-dd hh'),to_date(time,'yyyy-mm-ddhh'),'dd')=2 then 1 else 0 end)*datediff(to_date('${label_day}','yyyy-mm-ddhh'),to_date('2014-11-18 00','yyyy-mm-dd hh'),'dd')/sum(case whenbehavior_type=3 then 1 else 0 end) end as i11,
    case when sum(case when behavior_type=4then 1 else 0 end)=0 then 0 else sum(case when behavior_type=4 anddatediff(to_date('${label_day}','yyyy-mm-dd hh'),to_date(time,'yyyy-mm-ddhh'),'dd')=2 then 1 else 0 end)*datediff(to_date('${label_day}','yyyy-mm-ddhh'),to_date('2014-11-18 00','yyyy-mm-dd hh'),'dd')/sum(case whenbehavior_type=4 then 1 else 0 end) end as i12,
    --最近第3天的行为数与日平均行为数的比值
    case when sum(case when behavior_type=1then 1 else 0 end)=0 then 0 else sum(case when behavior_type=1 anddatediff(to_date('${label_day}','yyyy-mm-dd hh'),to_date(time,'yyyy-mm-ddhh'),'dd')=3 then 1 else 0 end)*datediff(to_date('${label_day}','yyyy-mm-ddhh'),to_date('2014-11-18 00','yyyy-mm-dd hh'),'dd')/sum(case whenbehavior_type=1 then 1 else 0 end) end as i13,
    case when sum(case when behavior_type=2then 1 else 0 end)=0 then 0 else sum(case when behavior_type=2 anddatediff(to_date('${label_day}','yyyy-mm-dd hh'),to_date(time,'yyyy-mm-ddhh'),'dd')=3 then 1 else 0 end)*datediff(to_date('${label_day}','yyyy-mm-ddhh'),to_date('2014-11-18 00','yyyy-mm-dd hh'),'dd')/sum(case whenbehavior_type=2 then 1 else 0 end) end as i14,
    case when sum(case when behavior_type=3then 1 else 0 end)=0 then 0 else sum(case when behavior_type=3 anddatediff(to_date('${label_day}','yyyy-mm-dd hh'),to_date(time,'yyyy-mm-ddhh'),'dd')=3 then 1 else 0 end)*datediff(to_date('${label_day}','yyyy-mm-ddhh'),to_date('2014-11-18 00','yyyy-mm-dd hh'),'dd')/sum(case whenbehavior_type=3 then 1 else 0 end) end as i15,
    case when sum(case when behavior_type=4then 1 else 0 end)=0 then 0 else sum(case when behavior_type=4 anddatediff(to_date('${label_day}','yyyy-mm-dd hh'),to_date(time,'yyyy-mm-ddhh'),'dd')=3 then 1 else 0 end)*datediff(to_date('${label_day}','yyyy-mm-ddhh'),to_date('2014-11-18 00','yyyy-mm-dd hh'),'dd')/sum(case whenbehavior_type=4 then 1 else 0 end) end as i16,
    --最近3天的行为数与日平均行为数的比值
    case when sum(case when behavior_type=1then 1 else 0 end)=0 then 0 else sum(case when behavior_type=1 anddatediff(to_date('${label_day}','yyyy-mm-dd hh'),to_date(time,'yyyy-mm-ddhh'),'dd')<4 then 1 else 0 end)*datediff(to_date('${label_day}','yyyy-mm-ddhh'),to_date('2014-11-18 00','yyyy-mm-dd hh'),'dd')/sum(case whenbehavior_type=1 then 1 else 0 end) end as i17,
    case when sum(case when behavior_type=2then 1 else 0 end)=0 then 0 else sum(case when behavior_type=2 anddatediff(to_date('${label_day}','yyyy-mm-dd hh'),to_date(time,'yyyy-mm-ddhh'),'dd')<4 then 1 else 0 end)*datediff(to_date('${label_day}','yyyy-mm-ddhh'),to_date('2014-11-18 00','yyyy-mm-dd hh'),'dd')/sum(case whenbehavior_type=2 then 1 else 0 end) end as i18,
    case when sum(case when behavior_type=3then 1 else 0 end)=0 then 0 else sum(case when behavior_type=3 anddatediff(to_date('${label_day}','yyyy-mm-dd hh'),to_date(time,'yyyy-mm-ddhh'),'dd')<4 then 1 else 0 end)*datediff(to_date('${label_day}','yyyy-mm-ddhh'),to_date('2014-11-18 00','yyyy-mm-dd hh'),'dd')/sum(case whenbehavior_type=3 then 1 else 0 end) end as i19,
    case when sum(case when behavior_type=4then 1 else 0 end)=0 then 0 else sum(case when behavior_type=4 anddatediff(to_date('${label_day}','yyyy-mm-dd hh'),to_date(time,'yyyy-mm-ddhh'),'dd')<4 then 1 else 0 end)*datediff(to_date('${label_day}','yyyy-mm-ddhh'),to_date('2014-11-18 00','yyyy-mm-dd hh'),'dd')/sum(case whenbehavior_type=4 then 1 else 0 end) end as i20
fromtianchi_lbs.tianchi_mobile_recommend_train_user
wheretime<'${label_day}'
group by item_id;
 
--已成功
--统计商品在类别中的排序(2015-5-15添加)
drop table if exists${table_label}_item_features_2;
create table${table_label}_item_features_2 as
select item_id,
    -- dense_rank() over 会返回次序
    dense_rank() over(partition by item_category order by ii1 desc) as i21,
    dense_rank() over(partition by item_category order by ii2 desc) as i22,
    dense_rank() over(partition by item_category order by ii3 desc) as i23,
    dense_rank() over(partition by item_category order by ii4 desc) as i24
from
(
    select item_category,item_id,
    sum(case when behavior_type=1 then 1 else 0 end) as ii1,
    sum(case when behavior_type=2 then 1 else 0 end) as ii2,
    sum(case when behavior_type=3 then 1 else 0 end) as ii3,
    sum(case when behavior_type=4 then 1 else 0 end) as ii4
    fromtianchi_lbs.tianchi_mobile_recommend_train_user
    where time<'${label_day}'
    group by item_id,item_category
)t;
 
#####################05-24
--商品交互的总人数（全部，最近1天，3天）
drop table if exists${table_label}_item_features_3;
create table${table_label}_item_features_3 as
select item_id,
--全部
sum(case when behavior_type=1 then 1 else 0 end) as i25,
sum(case when behavior_type=2 then 1 else 0 end) as i26,
sum(case when behavior_type=3 then 1 else 0 end) as i27,
sum(case when behavior_type=4 then 1 else 0 end) as i28,
--最近1天
sum(case when behavior_type=1 and datediff(to_date('${label_day}','yyyy-mm-ddhh'),to_date(time,'yyyy-mm-dd hh'),'dd')=1 then 1 else 0 end) as i29,
sum(case when behavior_type=2 and datediff(to_date('${label_day}','yyyy-mm-ddhh'),to_date(time,'yyyy-mm-dd hh'),'dd')=1 then 1 else 0 end) as i30,
sum(case when behavior_type=3 and datediff(to_date('${label_day}','yyyy-mm-ddhh'),to_date(time,'yyyy-mm-dd hh'),'dd')=1 then 1 else 0 end) as i31,
sum(case when behavior_type=4 and datediff(to_date('${label_day}','yyyy-mm-ddhh'),to_date(time,'yyyy-mm-dd hh'),'dd')=1 then 1 else 0 end) as i32,
--最近3天
sum(case when behavior_type=1 and datediff(to_date('${label_day}','yyyy-mm-ddhh'),to_date(time,'yyyy-mm-dd hh'),'dd')<4 then 1 else 0 end) as i33,
sum(case when behavior_type=2 and datediff(to_date('${label_day}','yyyy-mm-ddhh'),to_date(time,'yyyy-mm-dd hh'),'dd')<4 then 1 else 0 end) as i34,
sum(case when behavior_type=3 and datediff(to_date('${label_day}','yyyy-mm-ddhh'),to_date(time,'yyyy-mm-dd hh'),'dd')<4 then 1 else 0 end) as i35,
sum(case when behavior_type=4 and datediff(to_date('${label_day}','yyyy-mm-ddhh'),to_date(time,'yyyy-mm-dd hh'),'dd')<4 then 1 else 0 end) as i36
from
(
select distinct item_id,user_id,time,behavior_type
from tianchi_lbs.tianchi_mobile_recommend_train_user
where time<'${label_day}'
)t
group by item_id;
 
--商品行为数（最近1天，3天）
drop table if exists${table_label}_item_features_4;
create table${table_label}_item_features_4 as
select item_id,
sum(case when behavior_type=1 and datediff(to_date('${label_day}','yyyy-mm-ddhh'),to_date(time,'yyyy-mm-dd hh'),'dd')=1 then 1 else 0 end) as i37,
sum(case when behavior_type=2 and datediff(to_date('${label_day}','yyyy-mm-ddhh'),to_date(time,'yyyy-mm-dd hh'),'dd')=1 then 1 else 0 end) as i38,
sum(case when behavior_type=3 and datediff(to_date('${label_day}','yyyy-mm-ddhh'),to_date(time,'yyyy-mm-dd hh'),'dd')=1 then 1 else 0 end) as i39,
sum(case when behavior_type=4 and datediff(to_date('${label_day}','yyyy-mm-ddhh'),to_date(time,'yyyy-mm-dd hh'),'dd')=1 then 1 else 0 end) as i40,
sum(case when behavior_type=1 and datediff(to_date('${label_day}','yyyy-mm-ddhh'),to_date(time,'yyyy-mm-dd hh'),'dd')<4 then 1 else 0 end) as i41,
sum(case when behavior_type=2 and datediff(to_date('${label_day}','yyyy-mm-ddhh'),to_date(time,'yyyy-mm-dd hh'),'dd')<4 then 1 else 0 end) as i42,
sum(case when behavior_type=3 and datediff(to_date('${label_day}','yyyy-mm-ddhh'),to_date(time,'yyyy-mm-dd hh'),'dd')<4 then 1 else 0 end) as i43,
sum(case when behavior_type=4 and datediff(to_date('${label_day}','yyyy-mm-ddhh'),to_date(time,'yyyy-mm-dd hh'),'dd')<4 then 1 else 0 end) as i44
from tianchi_lbs.tianchi_mobile_recommend_train_user
where time<'${label_day}'
group by item_id;
 
--商品的购买转化率及转化率与类别平均转化率的比值
drop table if exists${table_label}_item_features_5;
create table${table_label}_item_features_5 as
select item_id,r1 asi45,r2 as i46,r3 as i47,
case when cr1>0 then r1/cr1 else 0 end as i48,
case when cr2>0 then r2/cr2 else 0 end as i49,
case when cr3>0 then r3/cr3 else 0 end as i50
from
(
select item_id,item_category,
case when sum(case when behavior_type=4 then 1 else 0 end)>0 then sum(case whenbehavior_type=1 then 1 else 0 end)/sum(case when behavior_type=4 then 1 else 0end) else 0 end as r1,
case when sum(case when behavior_type=4 then 1 else 0 end)>0 then sum(case whenbehavior_type=2 then 1 else 0 end)/sum(case when behavior_type=4 then 1 else 0end) else 0 end as r2,
case when sum(case when behavior_type=4 then 1 else 0 end)>0 then sum(case whenbehavior_type=3 then 1 else 0 end)/sum(case when behavior_type=4 then 1 else 0end) else 0 end as r3
from tianchi_lbs.tianchi_mobile_recommend_train_user
where time<'${label_day}'
group byitem_id,item_category
) t1
join
(
selectitem_category,
casewhen sum(case when behavior_type=4 then 1 else 0 end)>0 then sum(case whenbehavior_type=1 then 1 else 0 end)/sum(case when behavior_type=4 then 1 else 0end) else 0 end as cr1,
casewhen sum(case when behavior_type=4 then 1 else 0 end)>0 then sum(case whenbehavior_type=1 then 1 else 0 end)/sum(case when behavior_type=4 then 1 else 0end) else 0 end as cr2,
casewhen sum(case when behavior_type=4 then 1 else 0 end)>0 then sum(case whenbehavior_type=1 then 1 else 0 end)/sum(case when behavior_type=4 then 1 else 0end) else 0 end as cr3
fromtianchi_lbs.tianchi_mobile_recommend_train_user
wheretime<'${label_day}'
groupby item_category
) t2
ont1.item_category=t2.item_category;
 
--商品行为/同类同行为均值（总表，类别行为统计）
drop table if exists${table_label}_item_features_6;
create table${table_label}_item_features_6 as
select item_id,
casewhen t2.click>0 then t1.click/t2.click else 0 end as i51,
casewhen t2.favorite>0 then t1.favorite/t2.favorite else 0 end i52,
casewhen t2.cart>0 then t1.cart/t2.cart else 0 end i53,
casewhen t2.buy>0 then t1.buy/t2.buy else 0 end i54
from
(
--用户的行为数
selectitem_id,item_category,
sum(casewhen behavior_type=1 then 1 else 0 end) as click,
sum(casewhen behavior_type=2 then 1 else 0 end) as favorite,
sum(casewhen behavior_type=3 then 1 else 0 end) as cart,
sum(casewhen behavior_type=4 then 1 else 0 end) as buy
fromtianchi_lbs.tianchi_mobile_recommend_train_user
wheretime<'${label_day}'
groupby item_id,item_category
)t1
join
(
--类别的平均行为数
selectitem_category,
avg(casewhen behavior_type=1 then 1 else 0 end) as click,
avg(casewhen behavior_type=1 then 1 else 0 end) as favorite,
avg(casewhen behavior_type=1 then 1 else 0 end) as cart,
avg(casewhen behavior_type=1 then 1 else 0 end) as buy
fromtianchi_lbs.tianchi_mobile_recommend_train_user
wheretime<'${label_day}'
groupby item_category
)t2
ont1.item_category=t2.item_category;
 
--合并item特征。此处需要常常做变动，添加新的特征之后都需要做改变
drop table if exists${table_label}_item_features;
create table${table_label}_item_features as
select
   ${table_label}_item_features_1.item_id,i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12,i13,i14,i15,i16,i17,i18,i19,i20,i21,i22,i23,i24
,i25,i26,i27,i28,i29,i30,i31,i32,i33,i34,i35,i36,i37,i38,i39,i40,i41,i42,i43,i44,i45,i46,i47,i48,i49,i50,i51,i52,i53,i54
from
${table_label}_item_features_1
join
${table_label}_item_features_2
on${table_label}_item_features_1.item_id=${table_label}_item_features_2.item_id
join
${table_label}_item_features_3
on${table_label}_item_features_1.item_id=${table_label}_item_features_3.item_id
join
${table_label}_item_features_4
on${table_label}_item_features_1.item_id=${table_label}_item_features_4.item_id
join
${table_label}_item_features_5
on${table_label}_item_features_1.item_id=${table_label}_item_features_5.item_id
join
${table_label}_item_features_6
on${table_label}_item_features_1.item_id=${table_label}_item_features_6.item_id;
 
 
------------------------------------------------------------------------------
使用label_day之前的数据生成用户-商品特征
 
已成功
drop table if exists${table_label}_ui_features_1;
create table${table_label}_ui_features_1 as
SELECT
    user_id,item_id,
    --平均每天对商品的行为数
    sum(case when (behavior_type=1 andtime<'${label_day}') then 1 else 0 end)/30 as ui1,
    sum(case when (behavior_type=2 andtime<'${label_day}') then 1 else 0 end)/30 as ui2,
    sum(case when (behavior_type=3 andtime<'${label_day}') then 1 else 0 end)/30 as ui3,
    sum(case when (behavior_type=4 andtime<'${label_day}') then 1 else 0 end)/30 as ui4,
    --最近第一天的操作
    sum(case when (behavior_type=1 anddatediff(to_date('${label_day}','yyyy-mm-dd hh'),to_date(time,'yyyy-mm-ddhh'),'dd')=1) then 1 else 0 end) as ui5,
    sum(case when (behavior_type=2 anddatediff(to_date('${label_day}','yyyy-mm-dd hh'),to_date(time,'yyyy-mm-ddhh'),'dd')=1) then 1 else 0 end) as ui6,
    sum(case when (behavior_type=3 anddatediff(to_date('${label_day}','yyyy-mm-dd hh'),to_date(time,'yyyy-mm-ddhh'),'dd')=1) then 1 else 0 end) as ui7,
    --最近第二天的操作
    sum(case when (behavior_type=1 anddatediff(to_date('${label_day}','yyyy-mm-dd hh'),to_date(time,'yyyy-mm-ddhh'),'dd')=2) then 1 else 0 end) as ui8,
    sum(case when (behavior_type=2 anddatediff(to_date('${label_day}','yyyy-mm-dd hh'),to_date(time,'yyyy-mm-ddhh'),'dd')=2) then 1 else 0 end) as ui9,
    sum(case when (behavior_type=3 anddatediff(to_date('${label_day}','yyyy-mm-dd hh'),to_date(time,'yyyy-mm-ddhh'),'dd')=2) then 1 else 0 end) as ui10,
    sum(case when (behavior_type=4 anddatediff(to_date('${label_day}','yyyy-mm-dd hh'),to_date(time,'yyyy-mm-ddhh'),'dd')=2) then 1 else 0 end) as ui11,
    --最近第三天的操作
    sum(case when (behavior_type=1 anddatediff(to_date('${label_day}','yyyy-mm-dd hh'),to_date(time,'yyyy-mm-ddhh'),'dd')=3) then 1 else 0 end) as ui12,
    sum(case when (behavior_type=2 anddatediff(to_date('${label_day}','yyyy-mm-dd hh'),to_date(time,'yyyy-mm-ddhh'),'dd')=3) then 1 else 0 end) as ui13,
    sum(case when (behavior_type=3 anddatediff(to_date('${label_day}','yyyy-mm-dd hh'),to_date(time,'yyyy-mm-ddhh'),'dd')=3) then 1 else 0 end) as ui14,
    sum(case when (behavior_type=4 anddatediff(to_date('${label_day}','yyyy-mm-dd hh'),to_date(time,'yyyy-mm-ddhh'),'dd')=3) then 1 else 0 end) as ui15,
    --最近1周的操作
    sum(case when (behavior_type=1 anddatediff(to_date('${label_day}','yyyy-mm-dd hh'),to_date(time,'yyyy-mm-ddhh'),'dd')<8) then 1 else 0 end) as ui16,
    sum(case when (behavior_type=2 anddatediff(to_date('${label_day}','yyyy-mm-dd hh'),to_date(time,'yyyy-mm-ddhh'),'dd')<8) then 1 else 0 end) as ui17,
    sum(case when (behavior_type=3 anddatediff(to_date('${label_day}','yyyy-mm-dd hh'),to_date(time,'yyyy-mm-ddhh'),'dd')<8) then 1 else 0 end) as ui18,
    sum(case when (behavior_type=4 anddatediff(to_date('${label_day}','yyyy-mm-dd hh'),to_date(time,'yyyy-mm-ddhh'),'dd')<8) then 1 else 0 end) as ui19,
    --最近一天的最后的操作时间
    max(case when (behavior_type=1 anddatediff(to_date('${label_day}','yyyy-mm-dd hh'),to_date(time,'yyyy-mm-ddhh'),'dd')=1) then cast(substr(time,-2,2) as bigint) else 0 end) as ui20,
    max(case when (behavior_type=2 anddatediff(to_date('${label_day}','yyyy-mm-dd hh'),to_date(time,'yyyy-mm-ddhh'),'dd')=1) then cast(substr(time,-2,2) as bigint) else 0 end) as ui21,
    max(case when (behavior_type=3 anddatediff(to_date('${label_day}','yyyy-mm-dd hh'),to_date(time,'yyyy-mm-ddhh'),'dd')=1) then cast(substr(time,-2,2) as bigint) else 0 end) as ui22,
    --最近一天的最早的操作时间
    min(case when (behavior_type=1 anddatediff(to_date('${label_day}','yyyy-mm-dd hh'),to_date(time,'yyyy-mm-ddhh'),'dd')=1) then cast(substr(time,-2,2) as bigint) else 24 end) as ui23,
    min(case when (behavior_type=2 anddatediff(to_date('${label_day}','yyyy-mm-dd hh'),to_date(time,'yyyy-mm-ddhh'),'dd')=1) then cast(substr(time,-2,2) as bigint) else 24 end) as ui24,
    min(case when (behavior_type=3 anddatediff(to_date('${label_day}','yyyy-mm-dd hh'),to_date(time,'yyyy-mm-ddhh'),'dd')=1) then cast(substr(time,-2,2) as bigint) else 24 end) as ui25,
    --2015-5-16添加
    --用户最近的交互离现在的时间
    min(case when behavior_type=1 thendatediff(to_date('${label_day}','yyyy-mm-dd hh'),to_date(time,'yyyy-mm-ddhh'),'hh') else 744 end) as ui26,
    min(case when behavior_type=2 thendatediff(to_date('${label_day}','yyyy-mm-dd hh'),to_date(time,'yyyy-mm-ddhh'),'hh') else 744 end) as ui27,
    min(case when behavior_type=3 thendatediff(to_date('${label_day}','yyyy-mm-dd hh'),to_date(time,'yyyy-mm-ddhh'),'hh') else 744 end) as ui28,
    min(case when behavior_type=4 thendatediff(to_date('${label_day}','yyyy-mm-dd hh'),to_date(time,'yyyy-mm-ddhh'),'hh') else 744 end) as ui29,
    --用户商品交互的总天数
    count(distinct case when behavior_type=1then substr(time,7,4) end) as ui30,
    count(distinct case when behavior_type=4then substr(time,7,4) end) as ui31,
    --最近3天的操作
    sum(case when (behavior_type=1 anddatediff(to_date('${label_day}','yyyy-mm-dd hh'),to_date(time,'yyyy-mm-ddhh'),'dd')<4) then 1 else 0 end) as ui32,
    sum(case when (behavior_type=2 anddatediff(to_date('${label_day}','yyyy-mm-dd hh'),to_date(time,'yyyy-mm-ddhh'),'dd')<4) then 1 else 0 end) as ui33,
    sum(case when (behavior_type=3 anddatediff(to_date('${label_day}','yyyy-mm-dd hh'),to_date(time,'yyyy-mm-ddhh'),'dd')<4) then 1 else 0 end) as ui34,
    sum(case when (behavior_type=4 anddatediff(to_date('${label_day}','yyyy-mm-dd hh'),to_date(time,'yyyy-mm-ddhh'),'dd')<4) then 1 else 0 end) as ui35
FROMtianchi_lbs.tianchi_mobile_recommend_train_user
wheretime<'${label_day}'
GROUP BYuser_id,item_id;
 
2015-5-21添加
最近24小时用户的各种行为数（4*24）
drop table if exists${table_label}_ui_features_2;
create table${table_label}_ui_features_2 as
selectuser_id,item_id,
sum(casewhen behavior_type=1 and datediff(to_date('${label_day}','yyyy-mm-ddhh'),to_date(time,'yyyy-mm-dd hh'),'hh')=24 then 1 else 0 end) as ui36,
sum(casewhen behavior_type=1 and datediff(to_date('${label_day}','yyyy-mm-ddhh'),to_date(time,'yyyy-mm-dd hh'),'hh')=23 then 1 else 0 end) as ui37,
sum(casewhen behavior_type=1 and datediff(to_date('${label_day}','yyyy-mm-ddhh'),to_date(time,'yyyy-mm-dd hh'),'hh')=22 then 1 else 0 end) as ui38,
sum(casewhen behavior_type=1 and datediff(to_date('${label_day}','yyyy-mm-ddhh'),to_date(time,'yyyy-mm-dd hh'),'hh')=21 then 1 else 0 end) as ui39,
sum(casewhen behavior_type=1 and datediff(to_date('${label_day}','yyyy-mm-ddhh'),to_date(time,'yyyy-mm-dd hh'),'hh')=20 then 1 else 0 end) as ui40,
sum(casewhen behavior_type=1 and datediff(to_date('${label_day}','yyyy-mm-ddhh'),to_date(time,'yyyy-mm-dd hh'),'hh')=19 then 1 else 0 end) as ui41,
sum(casewhen behavior_type=1 and datediff(to_date('${label_day}','yyyy-mm-ddhh'),to_date(time,'yyyy-mm-dd hh'),'hh')=18 then 1 else 0 end) as ui42,
sum(casewhen behavior_type=1 and datediff(to_date('${label_day}','yyyy-mm-ddhh'),to_date(time,'yyyy-mm-dd hh'),'hh')=17 then 1 else 0 end) as ui43,
sum(casewhen behavior_type=1 and datediff(to_date('${label_day}','yyyy-mm-ddhh'),to_date(time,'yyyy-mm-dd hh'),'hh')=16 then 1 else 0 end) as ui44,
sum(casewhen behavior_type=1 and datediff(to_date('${label_day}','yyyy-mm-ddhh'),to_date(time,'yyyy-mm-dd hh'),'hh')=15 then 1 else 0 end) as ui45,
sum(casewhen behavior_type=1 and datediff(to_date('${label_day}','yyyy-mm-ddhh'),to_date(time,'yyyy-mm-dd hh'),'hh')=14 then 1 else 0 end) as ui46,
sum(casewhen behavior_type=1 and datediff(to_date('${label_day}','yyyy-mm-ddhh'),to_date(time,'yyyy-mm-dd hh'),'hh')=13 then 1 else 0 end) as ui47,
sum(casewhen behavior_type=1 and datediff(to_date('${label_day}','yyyy-mm-ddhh'),to_date(time,'yyyy-mm-dd hh'),'hh')=12 then 1 else 0 end) as ui48,
sum(casewhen behavior_type=1 and datediff(to_date('${label_day}','yyyy-mm-ddhh'),to_date(time,'yyyy-mm-dd hh'),'hh')=11 then 1 else 0 end) as ui49,
sum(casewhen behavior_type=1 and datediff(to_date('${label_day}','yyyy-mm-ddhh'),to_date(time,'yyyy-mm-dd hh'),'hh')=10 then 1 else 0 end) as ui50,
sum(casewhen behavior_type=1 and datediff(to_date('${label_day}','yyyy-mm-ddhh'),to_date(time,'yyyy-mm-dd hh'),'hh')=9 then 1 else 0 end) as ui51,
sum(casewhen behavior_type=1 and datediff(to_date('${label_day}','yyyy-mm-ddhh'),to_date(time,'yyyy-mm-dd hh'),'hh')=8 then 1 else 0 end) as ui52,
sum(casewhen behavior_type=1 and datediff(to_date('${label_day}','yyyy-mm-ddhh'),to_date(time,'yyyy-mm-dd hh'),'hh')=7 then 1 else 0 end) as ui53,
sum(casewhen behavior_type=1 and datediff(to_date('${label_day}','yyyy-mm-ddhh'),to_date(time,'yyyy-mm-dd hh'),'hh')=6 then 1 else 0 end) as ui54,
sum(casewhen behavior_type=1 and datediff(to_date('${label_day}','yyyy-mm-ddhh'),to_date(time,'yyyy-mm-dd hh'),'hh')=5 then 1 else 0 end) as ui55,
sum(casewhen behavior_type=1 and datediff(to_date('${label_day}','yyyy-mm-ddhh'),to_date(time,'yyyy-mm-dd hh'),'hh')=4 then 1 else 0 end) as ui56,
sum(casewhen behavior_type=1 and datediff(to_date('${label_day}','yyyy-mm-ddhh'),to_date(time,'yyyy-mm-dd hh'),'hh')=3 then 1 else 0 end) as ui57,
sum(casewhen behavior_type=1 and datediff(to_date('${label_day}','yyyy-mm-ddhh'),to_date(time,'yyyy-mm-dd hh'),'hh')=2 then 1 else 0 end) as ui58,
sum(casewhen behavior_type=1 and datediff(to_date('${label_day}','yyyy-mm-ddhh'),to_date(time,'yyyy-mm-dd hh'),'hh')=1 then 1 else 0 end) as ui59,
sum(casewhen behavior_type=2 and datediff(to_date('${label_day}','yyyy-mm-ddhh'),to_date(time,'yyyy-mm-dd hh'),'hh')=24 then 1 else 0 end) as ui60,
sum(casewhen behavior_type=2 and datediff(to_date('${label_day}','yyyy-mm-ddhh'),to_date(time,'yyyy-mm-dd hh'),'hh')=23 then 1 else 0 end) as ui61,
sum(casewhen behavior_type=2 and datediff(to_date('${label_day}','yyyy-mm-ddhh'),to_date(time,'yyyy-mm-dd hh'),'hh')=22 then 1 else 0 end) as ui62,
sum(casewhen behavior_type=2 and datediff(to_date('${label_day}','yyyy-mm-ddhh'),to_date(time,'yyyy-mm-dd hh'),'hh')=21 then 1 else 0 end) as ui63,
sum(casewhen behavior_type=2 and datediff(to_date('${label_day}','yyyy-mm-ddhh'),to_date(time,'yyyy-mm-dd hh'),'hh')=20 then 1 else 0 end) as ui64,
sum(casewhen behavior_type=2 and datediff(to_date('${label_day}','yyyy-mm-ddhh'),to_date(time,'yyyy-mm-dd hh'),'hh')=19 then 1 else 0 end) as ui65,
sum(casewhen behavior_type=2 and datediff(to_date('${label_day}','yyyy-mm-ddhh'),to_date(time,'yyyy-mm-dd hh'),'hh')=18 then 1 else 0 end) as ui66,
sum(casewhen behavior_type=2 and datediff(to_date('${label_day}','yyyy-mm-ddhh'),to_date(time,'yyyy-mm-dd hh'),'hh')=17 then 1 else 0 end) as ui67,
sum(casewhen behavior_type=2 and datediff(to_date('${label_day}','yyyy-mm-ddhh'),to_date(time,'yyyy-mm-dd hh'),'hh')=16 then 1 else 0 end) as ui68,
sum(casewhen behavior_type=2 and datediff(to_date('${label_day}','yyyy-mm-ddhh'),to_date(time,'yyyy-mm-dd hh'),'hh')=15 then 1 else 0 end) as ui69,
sum(casewhen behavior_type=2 and datediff(to_date('${label_day}','yyyy-mm-ddhh'),to_date(time,'yyyy-mm-dd hh'),'hh')=14 then 1 else 0 end) as ui70,
sum(casewhen behavior_type=2 and datediff(to_date('${label_day}','yyyy-mm-ddhh'),to_date(time,'yyyy-mm-dd hh'),'hh')=13 then 1 else 0 end) as ui71,
sum(casewhen behavior_type=2 and datediff(to_date('${label_day}','yyyy-mm-ddhh'),to_date(time,'yyyy-mm-dd hh'),'hh')=12 then 1 else 0 end) as ui72,
sum(casewhen behavior_type=2 and datediff(to_date('${label_day}','yyyy-mm-ddhh'),to_date(time,'yyyy-mm-dd hh'),'hh')=11 then 1 else 0 end) as ui73,
sum(casewhen behavior_type=2 and datediff(to_date('${label_day}','yyyy-mm-ddhh'),to_date(time,'yyyy-mm-dd hh'),'hh')=10 then 1 else 0 end) as ui74,
sum(casewhen behavior_type=2 and datediff(to_date('${label_day}','yyyy-mm-ddhh'),to_date(time,'yyyy-mm-dd hh'),'hh')=9 then 1 else 0 end) as ui75,
sum(casewhen behavior_type=2 and datediff(to_date('${label_day}','yyyy-mm-ddhh'),to_date(time,'yyyy-mm-dd hh'),'hh')=8 then 1 else 0 end) as ui76,
sum(casewhen behavior_type=2 and datediff(to_date('${label_day}','yyyy-mm-ddhh'),to_date(time,'yyyy-mm-dd hh'),'hh')=7 then 1 else 0 end) as ui77,
sum(casewhen behavior_type=2 and datediff(to_date('${label_day}','yyyy-mm-ddhh'),to_date(time,'yyyy-mm-dd hh'),'hh')=6 then 1 else 0 end) as ui78,
sum(casewhen behavior_type=2 and datediff(to_date('${label_day}','yyyy-mm-ddhh'),to_date(time,'yyyy-mm-dd hh'),'hh')=5 then 1 else 0 end) as ui79,
sum(casewhen behavior_type=2 and datediff(to_date('${label_day}','yyyy-mm-ddhh'),to_date(time,'yyyy-mm-dd hh'),'hh')=4 then 1 else 0 end) as ui80,
sum(casewhen behavior_type=2 and datediff(to_date('${label_day}','yyyy-mm-ddhh'),to_date(time,'yyyy-mm-dd hh'),'hh')=3 then 1 else 0 end) as ui81,
sum(casewhen behavior_type=2 and datediff(to_date('${label_day}','yyyy-mm-ddhh'),to_date(time,'yyyy-mm-dd hh'),'hh')=2 then 1 else 0 end) as ui82,
sum(casewhen behavior_type=2 and datediff(to_date('${label_day}','yyyy-mm-ddhh'),to_date(time,'yyyy-mm-dd hh'),'hh')=1 then 1 else 0 end) as ui83,
sum(casewhen behavior_type=3 and datediff(to_date('${label_day}','yyyy-mm-ddhh'),to_date(time,'yyyy-mm-dd hh'),'hh')=24 then 1 else 0 end) as ui84,
sum(casewhen behavior_type=3 and datediff(to_date('${label_day}','yyyy-mm-ddhh'),to_date(time,'yyyy-mm-dd hh'),'hh')=23 then 1 else 0 end) as ui85,
sum(casewhen behavior_type=3 and datediff(to_date('${label_day}','yyyy-mm-ddhh'),to_date(time,'yyyy-mm-dd hh'),'hh')=22 then 1 else 0 end) as ui86,
sum(casewhen behavior_type=3 and datediff(to_date('${label_day}','yyyy-mm-ddhh'),to_date(time,'yyyy-mm-dd hh'),'hh')=21 then 1 else 0 end) as ui87,
sum(casewhen behavior_type=3 and datediff(to_date('${label_day}','yyyy-mm-ddhh'),to_date(time,'yyyy-mm-dd hh'),'hh')=20 then 1 else 0 end) as ui88,
sum(casewhen behavior_type=3 and datediff(to_date('${label_day}','yyyy-mm-ddhh'),to_date(time,'yyyy-mm-dd hh'),'hh')=19 then 1 else 0 end) as ui89,
sum(casewhen behavior_type=3 and datediff(to_date('${label_day}','yyyy-mm-ddhh'),to_date(time,'yyyy-mm-dd hh'),'hh')=18 then 1 else 0 end) as ui90,
sum(casewhen behavior_type=3 and datediff(to_date('${label_day}','yyyy-mm-ddhh'),to_date(time,'yyyy-mm-dd hh'),'hh')=17 then 1 else 0 end) as ui91,
sum(casewhen behavior_type=3 and datediff(to_date('${label_day}','yyyy-mm-ddhh'),to_date(time,'yyyy-mm-dd hh'),'hh')=16 then 1 else 0 end) as ui92,
sum(casewhen behavior_type=3 and datediff(to_date('${label_day}','yyyy-mm-ddhh'),to_date(time,'yyyy-mm-dd hh'),'hh')=15 then 1 else 0 end) as ui93,
sum(casewhen behavior_type=3 and datediff(to_date('${label_day}','yyyy-mm-ddhh'),to_date(time,'yyyy-mm-dd hh'),'hh')=14 then 1 else 0 end) as ui94,
sum(casewhen behavior_type=3 and datediff(to_date('${label_day}','yyyy-mm-ddhh'),to_date(time,'yyyy-mm-dd hh'),'hh')=13 then 1 else 0 end) as ui95,
sum(casewhen behavior_type=3 and datediff(to_date('${label_day}','yyyy-mm-ddhh'),to_date(time,'yyyy-mm-dd hh'),'hh')=12 then 1 else 0 end) as ui96,
sum(casewhen behavior_type=3 and datediff(to_date('${label_day}','yyyy-mm-ddhh'),to_date(time,'yyyy-mm-dd hh'),'hh')=11 then 1 else 0 end) as ui97,
sum(casewhen behavior_type=3 and datediff(to_date('${label_day}','yyyy-mm-ddhh'),to_date(time,'yyyy-mm-dd hh'),'hh')=10 then 1 else 0 end) as ui98,
sum(casewhen behavior_type=3 and datediff(to_date('${label_day}','yyyy-mm-ddhh'),to_date(time,'yyyy-mm-dd hh'),'hh')=9 then 1 else 0 end) as ui99,
sum(casewhen behavior_type=3 and datediff(to_date('${label_day}','yyyy-mm-ddhh'),to_date(time,'yyyy-mm-dd hh'),'hh')=8 then 1 else 0 end) as ui100,
sum(casewhen behavior_type=3 and datediff(to_date('${label_day}','yyyy-mm-ddhh'),to_date(time,'yyyy-mm-dd hh'),'hh')=7 then 1 else 0 end) as ui101,
sum(casewhen behavior_type=3 and datediff(to_date('${label_day}','yyyy-mm-ddhh'),to_date(time,'yyyy-mm-dd hh'),'hh')=6 then 1 else 0 end) as ui102,
sum(casewhen behavior_type=3 and datediff(to_date('${label_day}','yyyy-mm-ddhh'),to_date(time,'yyyy-mm-dd hh'),'hh')=5 then 1 else 0 end) as ui103,
sum(casewhen behavior_type=3 and datediff(to_date('${label_day}','yyyy-mm-ddhh'),to_date(time,'yyyy-mm-dd hh'),'hh')=4 then 1 else 0 end) as ui104,
sum(casewhen behavior_type=3 and datediff(to_date('${label_day}','yyyy-mm-ddhh'),to_date(time,'yyyy-mm-dd hh'),'hh')=3 then 1 else 0 end) as ui105,
sum(casewhen behavior_type=3 and datediff(to_date('${label_day}','yyyy-mm-ddhh'),to_date(time,'yyyy-mm-dd hh'),'hh')=2 then 1 else 0 end) as ui106,
sum(casewhen behavior_type=3 and datediff(to_date('${label_day}','yyyy-mm-ddhh'),to_date(time,'yyyy-mm-dd hh'),'hh')=1 then 1 else 0 end) as ui107
fromtianchi_lbs.tianchi_mobile_recommend_train_user
wheredatediff(to_date('${label_day}','yyyy-mm-dd hh'),to_date(time,'yyyy-mm-ddhh'),'dd')=1
group byuser_id,item_id;
 
--用户A对品牌B的总购买数\收藏数\购物车数（总表,测试通过）
--用户A对品牌B的点击数的平方（总表）
--用户A对品牌B的购买数的平方（总表）
--用户A对品牌B的点击购买比（可用总表替代）
drop table if exists${table_label}_ui_features_2;
create table${table_label}_ui_features_2 as
selectuser_id,item_id,
sum(casewhen behavior_type=1 then 1 else 0 end) as ui36,
sum(casewhen behavior_type=2 then 1 else 0 end) as ui37,
sum(casewhen behavior_type=3 then 1 else 0 end) as ui38,
sum(casewhen behavior_type=4 then 1 else 0 end) as ui39,
sum(casewhen behavior_type=1 then 1 else 0 end)*sum(case when behavior_type=1 then 1else 0 end) as ui40,
sum(casewhen behavior_type=2 then 1 else 0 end)*sum(case when behavior_type=2 then 1else 0 end) as ui41,
sum(casewhen behavior_type=3 then 1 else 0 end)*sum(case when behavior_type=2 then 1else 0 end) as ui42,
sum(casewhen behavior_type=4 then 1 else 0 end)*sum(case when behavior_type=4 then 1else 0 end) as ui43,
casewhen sum(case when behavior_type=4 then 1 else 0 end)>0 then sum(case whenbehavior_type=1 then 1 else 0 end)/sum(case when behavior_type=4 then 1 else 0end) else 0 end as ui44,
casewhen sum(case when behavior_type=4 then 1 else 0 end)>0 then sum(case whenbehavior_type=2 then 1 else 0 end)/sum(case when behavior_type=4 then 1 else 0end) else 0 end as ui45,
casewhen sum(case when behavior_type=4 then 1 else 0 end)>0 then sum(case whenbehavior_type=3 then 1 else 0 end)/sum(case when behavior_type=4 then 1 else 0end) else 0 end as ui46,
casewhen sum(case when behavior_type=4 then 1 else 0 end)>0 then sum(case whenbehavior_type=4 then 1 else 0 end)/sum(case when behavior_type=4 then 1 else 0end) else 0 end as ui47
fromtianchi_lbs.tianchi_mobile_recommend_train_user
wheretime<'${label_day}'
group byuser_id,item_id;
 
--测试通过
--用户交互本商品前，交互的商品数（4）（时间上为>=）
--用户交互本商品后，交互的商品数（4）（时间上为<=）
drop table if exists${table_label}_ui_features_3;
create table${table_label}_ui_features_3 as
selectt1.user_id,t1.item_id,
sum(casewhen t1.behavior_type=1 and t2.behavior_type=1 and t2.hour<=t1.s then 1 else0 end) as ui48,
sum(casewhen t1.behavior_type=2 and t2.behavior_type=2 and t2.hour<=t1.s then 1 else0 end) as ui49,
sum(casewhen t1.behavior_type=3 and t2.behavior_type=3 and t2.hour<=t1.s then 1 else0 end) as ui50,
sum(casewhen t2.behavior_type=4 and t2.hour<=t1.s then 1 else 0 end) as ui51,
sum(casewhen t1.behavior_type=1 and t2.behavior_type=1 and t2.hour>t1.e then 1 else0 end) as ui52,
sum(casewhen t1.behavior_type=2 and t2.behavior_type=1 and t2.hour>t1.e then 1 else0 end) as ui53,
sum(casewhen t1.behavior_type=3 and t2.behavior_type=1 and t2.hour>t1.e then 1 else0 end) as ui54,
sum(casewhen t2.behavior_type=4 and t2.hour>t1.e then 1 else 0 end) as ui55
from
(--前一天的交互行为最早最晚时间
selectuser_id,item_id,behavior_type,min(cast(substr(time,-2,2) as bigint)) ass,max(cast(substr(time,-2,2) as bigint)) as e
fromtianchi_lbs.tianchi_mobile_recommend_train_user
wheredatediff(to_date('${label_day}','yyyy-mm-dd hh'),to_date(time,'yyyy-mm-ddhh'),'dd')=1
groupby user_id,item_id,behavior_type
) t1
join
--最近一天用户与商品的交互行为
(
selectdistinct user_id,item_id,behavior_type,cast(substr(time,-2,2) as bigint) ashour
fromtianchi_lbs.tianchi_mobile_recommend_train_user
wheredatediff(to_date('${label_day}','yyyy-mm-dd hh'),to_date(time,'yyyy-mm-ddhh'),'dd')=1
) t2
ont1.user_id=t2.user_id
group byt1.user_id,t1.item_id;
 
--合并ui特征，这里需要常常改动
drop table if exists${table_label}_ui_features;
create table${table_label}_ui_features as
SELECT
   ${table_label}_ui_features_1.user_id,${table_label}_ui_features_1.item_id,
   ui1,ui2,ui3,ui4,ui5,ui6,ui7,ui8,ui9,ui10,ui11,ui12,ui13,ui14,ui15,ui16,ui17,ui18,ui19,ui20,ui21,ui22,ui23,ui24,ui25,ui26,ui27,ui28,ui29,ui30,ui31,ui32,ui33,ui34,ui35
    --ui_features_2
    ,case when ui36 is not null then ui36 else0 end as ui36
,casewhen ui37 is not null then ui37 else 0 end as ui37
,casewhen ui38 is not null then ui38 else 0 end as ui38
,casewhen ui39 is not null then ui39 else 0 end as ui39
,casewhen ui40 is not null then ui40 else 0 end as ui40
,casewhen ui41 is not null then ui41 else 0 end as ui41
,casewhen ui42 is not null then ui42 else 0 end as ui42
,casewhen ui43 is not null then ui43 else 0 end as ui43
,casewhen ui44 is not null then ui44 else 0 end as ui44
,casewhen ui45 is not null then ui45 else 0 end as ui45
,casewhen ui46 is not null then ui46 else 0 end as ui46
,casewhen ui47 is not null then ui47 else 0 end as ui47
,casewhen ui48 is not null then ui48 else 0 end as ui48
,casewhen ui49 is not null then ui49 else 0 end as ui49
,casewhen ui50 is not null then ui50 else 0 end as ui50
,casewhen ui51 is not null then ui51 else 0 end as ui51
,casewhen ui52 is not null then ui52 else 0 end as ui52
,casewhen ui53 is not null then ui53 else 0 end as ui53
,casewhen ui54 is not null then ui54 else 0 end as ui54
,casewhen ui55 is not null then ui55 else 0 end as ui55
FROM${table_label}_ui_features_1
left outer join${table_label}_ui_features_2
on${table_label}_ui_features_1.user_id=${table_label}_ui_features_2.user_id and${table_label}_ui_features_1.item_id=${table_label}_ui_features_2.item_id
left outer join${table_label}_ui_features_3
on${table_label}_ui_features_1.user_id=${table_label}_ui_features_3.user_id and${table_label}_ui_features_1.item_id=${table_label}_ui_features_3.item_id;
 
使用label_day之前的数据生成user特征
 
已成功
drop table if exists${table_label}_user_features_1;
create table${table_label}_user_features_1 as
select user_id,
    --最近1天用户行为统计
    sum(case when (behavior_type=1 anddatediff(to_date('${label_day}','yyyy-mm-dd hh'),to_date(time,'yyyy-mm-ddhh'),'dd')=1) then 1 else 0 end) as u1,
    sum(case when (behavior_type=2 anddatediff(to_date('${label_day}','yyyy-mm-dd hh'),to_date(time,'yyyy-mm-ddhh'),'dd')=1) then 1 else 0 end) as u2,
    sum(case when (behavior_type=3 anddatediff(to_date('${label_day}','yyyy-mm-dd hh'),to_date(time,'yyyy-mm-ddhh'),'dd')=1) then 1 else 0 end) as u3,
    sum(case when (behavior_type=4 anddatediff(to_date('${label_day}','yyyy-mm-dd hh'),to_date(time,'yyyy-mm-ddhh'),'dd')=1) then 1 else 0 end) as u4,
    --最近3天用户行为统计
    sum(case when (behavior_type=1 anddatediff(to_date('${label_day}','yyyy-mm-dd hh'),to_date(time,'yyyy-mm-ddhh'),'dd')<4 and datediff(to_date('${label_day}','yyyy-mm-ddhh'),to_date(time,'yyyy-mm-dd hh'),'dd')>0) then 1 else 0 end) as u5,
    sum(case when (behavior_type=2 anddatediff(to_date('${label_day}','yyyy-mm-dd hh'),to_date(time,'yyyy-mm-ddhh'),'dd')<4 and datediff(to_date('${label_day}','yyyy-mm-ddhh'),to_date(time,'yyyy-mm-dd hh'),'dd')>0) then 1 else 0 end) as u6,
    sum(case when (behavior_type=3 anddatediff(to_date('${label_day}','yyyy-mm-dd hh'),to_date(time,'yyyy-mm-ddhh'),'dd')<4 and datediff(to_date('${label_day}','yyyy-mm-ddhh'),to_date(time,'yyyy-mm-dd hh'),'dd')>0) then 1 else 0 end) as u7,
    sum(case when (behavior_type=4 anddatediff(to_date('${label_day}','yyyy-mm-dd hh'),to_date(time,'yyyy-mm-ddhh'),'dd')<4 and datediff(to_date('${label_day}','yyyy-mm-ddhh'),to_date(time,'yyyy-mm-dd hh'),'dd')>0) then 1 else 0 end) as u8,
    --过去全部的用户行为统计
    sum(case when (behavior_type=1 andtime<'${label_day}') then 1 else 0 end) as u9,
    sum(case when (behavior_type=2 andtime<'${label_day}') then 1 else 0 end) as u10,
    sum(case when (behavior_type=3 andtime<'${label_day}') then 1 else 0 end) as u11,
    sum(case when (behavior_type=4 andtime<'${label_day}') then 1 else 0 end) as u12,
    --2015-5-15添加
    --用户前一天最早的交互行为时间
    min(case when (behavior_type=1 anddatediff(to_date('${label_day}','yyyy-mm-dd hh'),to_date(time,'yyyy-mm-ddhh'),'dd')=1) then cast(substr(time,-2,2) as bigint) else 24 end) as u13,
    min(case when (behavior_type=2 anddatediff(to_date('${label_day}','yyyy-mm-dd hh'),to_date(time,'yyyy-mm-ddhh'),'dd')=1) then cast(substr(time,-2,2) as bigint) else 24 end) as u14,
    min(case when (behavior_type=3 anddatediff(to_date('${label_day}','yyyy-mm-dd hh'),to_date(time,'yyyy-mm-ddhh'),'dd')=1) then cast(substr(time,-2,2) as bigint) else 24 end) as u15,
    --用户前一天最晚的交互行为时间
    max(case when (behavior_type=1 anddatediff(to_date('${label_day}','yyyy-mm-dd hh'),to_date(time,'yyyy-mm-ddhh'),'dd')=1) then cast(substr(time,-2,2) as bigint) else 0 end) as u16,
    max(case when (behavior_type=2 anddatediff(to_date('${label_day}','yyyy-mm-dd hh'),to_date(time,'yyyy-mm-ddhh'),'dd')=1) then cast(substr(time,-2,2) as bigint) else 0 end) as u17,
    max(case when (behavior_type=3 anddatediff(to_date('${label_day}','yyyy-mm-dd hh'),to_date(time,'yyyy-mm-ddhh'),'dd')=1) then cast(substr(time,-2,2) as bigint) else 0 end) as u18,
    --最近一天用户交互了多少种item
    count(distinct case when behavior_type=1and datediff(to_date('${label_day}','yyyy-mm-dd hh'),to_date(time,'yyyy-mm-ddhh'),'dd')=1 then item_id end) as u19,
    count(distinct case when behavior_type=2and datediff(to_date('${label_day}','yyyy-mm-dd hh'),to_date(time,'yyyy-mm-ddhh'),'dd')=1 then item_id end) as u20,
    count(distinct case when behavior_type=3and datediff(to_date('${label_day}','yyyy-mm-dd hh'),to_date(time,'yyyy-mm-ddhh'),'dd')=1 then item_id end) as u21,
    count(distinct case when behavior_type=4and datediff(to_date('${label_day}','yyyy-mm-dd hh'),to_date(time,'yyyy-mm-ddhh'),'dd')=1 then item_id end) as u22
fromtianchi_lbs.tianchi_mobile_recommend_train_user
wheretime<'${label_day}'
group by user_id;
 
--------------------------------------------------------------------------------------------------
--最近1天用户交互了多少种类别(测试通过)
drop table if exists${table_label}_user_features_2;
create table${table_label}_user_features_2 as
selectuser_id,count(item_category) as u23 from
(
selectdistinct user_id,item_category
fromtianchi_lbs.tianchi_mobile_recommend_train_user
wheredatediff(to_date('${label_day}','yyyy-mm-dd hh'),to_date(time,'yyyy-mm-ddhh'),'dd')=1
)t
group by user_id;
 
--用户的购买转化率(测试通过)
drop table if exists${table_label}_user_features_3;
create table${table_label}_user_features_3 as
select user_id,
casewhen sum(case when behavior_type=4 then 1 else 0 end)>0 then sum(case whenbehavior_type=1 then 1 else 0 end)/sum(case when behavior_type=4 then 1 else 0end) else 0 end as u24,
casewhen sum(case when behavior_type=4 then 1 else 0 end)>0 then sum(case whenbehavior_type=2 then 1 else 0 end)/sum(case when behavior_type=4 then 1 else 0end) else 0 end as u25,
casewhen sum(case when behavior_type=4 then 1 else 0 end)>0 then sum(case whenbehavior_type=3 then 1 else 0 end)/sum(case when behavior_type=4 then 1 else 0end) else 0 end as u26
fromtianchi_lbs.tianchi_mobile_recommend_train_user
wheretime<'${label_day}'
group by user_id;
 
--用户对交互过的商品的平均交互数，最大交互数，最小交互数(测试通过)
drop table if exists${table_label}_user_features_4;
create table${table_label}_user_features_4 as
select user_id,
max(cli)as u27,
min(cli)as u28,
avg(cli)as u29,
max(fav)as u30,
min(fav)as u31,
avg(fav)as u32,
max(car)as u33,
min(car)as u34,
avg(car)as u35,
max(buy)as u36,
min(buy)as u37,
avg(buy)as u38
from
(
selectuser_id,item_id,
sum(casewhen behavior_type=1 then 1 else 0 end) as cli,
sum(casewhen behavior_type=2 then 1 else 0 end) as fav,
sum(casewhen behavior_type=3 then 1 else 0 end) as car,
sum(casewhen behavior_type=4 then 1 else 0 end) as buy
fromtianchi_lbs.tianchi_mobile_recommend_train_user
wheretime<'${label_day}'
groupby user_id,item_id
) t
group by user_id;
 
--用户购买商品的时间（平均，最早，最晚）#生活习惯(测试通过)
drop table if exists${table_label}_user_features_5;
create table${table_label}_user_features_5 as
select user_id,
max(cast(substr(time,-2,2)as bigint)) as u39,
min(cast(substr(time,-2,2)as bigint)) as u40,
avg(cast(substr(time,-2,2)as bigint)) as u41
fromtianchi_lbs.tianchi_mobile_recommend_train_user
wheretime<'${label_day}'
group by user_id;
 
--用户有交互的天数(测试通过)
drop table if exists${table_label}_user_features_6;
create table${table_label}_user_features_6 as
select user_id,
sum(casewhen behavior_type=1 then 1 else 0 end) as u42,
sum(casewhen behavior_type=2 then 1 else 0 end) as u43,
sum(casewhen behavior_type=3 then 1 else 0 end) as u44,
sum(casewhen behavior_type=4 then 1 else 0 end) as u45
from
(
selectdistinct user_id,behavior_type,substr(time,9,2) as daynum
fromtianchi_lbs.tianchi_mobile_recommend_train_user
wheretime<'${label_day}'
 ) t
group by user_id;
 
--合并user特征
drop table if exists${table_label}_user_features;
create table${table_label}_user_features as
select
   ${table_label}_user_features_1.user_id,u1,u2,u3,u4,u5,u6,u7,u8,u9,u10,u11,u12,u13,u14,u15,u16,u17,u18,u19,u20,u21,u22
,casewhen u23 is not null then u23 else 0 end as u23
,casewhen u24 is not null then u24 else 0 end as u24
,casewhen u25 is not null then u25 else 0 end as u25
,casewhen u26 is not null then u26 else 0 end as u26
,casewhen u27 is not null then u27 else 0 end as u27
,casewhen u28 is not null then u28 else 0 end as u28
,casewhen u29 is not null then u29 else 0 end as u29
,casewhen u30 is not null then u30 else 0 end as u30
,casewhen u31 is not null then u31 else 0 end as u31
,casewhen u32 is not null then u32 else 0 end as u32
,casewhen u33 is not null then u33 else 0 end as u33
,casewhen u34 is not null then u34 else 0 end as u34
,casewhen u35 is not null then u35 else 0 end as u35
,casewhen u36 is not null then u36 else 0 end as u36
,casewhen u37 is not null then u37 else 0 end as u37
,casewhen u38 is not null then u38 else 0 end as u38
,casewhen u39 is not null then u39 else 0 end as u39
,casewhen u40 is not null then u40 else 0 end as u40
,casewhen u41 is not null then u41 else 0 end as u41
,casewhen u42 is not null then u42 else 0 end as u42
,casewhen u43 is not null then u43 else 0 end as u43
,casewhen u44 is not null then u44 else 0 end as u44
,casewhen u45 is not null then u45 else 0 end as u45
from${table_label}_user_features_1
left outer join${table_label}_user_features_2
on${table_label}_user_features_1.user_id=${table_label}_user_features_2.user_id
left outer join${table_label}_user_features_3
on${table_label}_user_features_1.user_id=${table_label}_user_features_3.user_id
left outer join${table_label}_user_features_4
on${table_label}_user_features_1.user_id=${table_label}_user_features_4.user_id
left outer join${table_label}_user_features_5
on${table_label}_user_features_1.user_id=${table_label}_user_features_5.user_id
left outer join${table_label}_user_features_6
on${table_label}_user_features_1.user_id=${table_label}_user_features_6.user_id;
 
--------------------------------------------------------------------------------
--                              分类标签                                     --
--------------------------------------------------------------------------------
 
--用于预测的作用对（label_day的前两天有交互且无购买）
--这个相当于模型之前的规则过滤，是一个常常需要调整的地方
drop table if existstemp_pairs_table;
create tabletemp_pairs_table as
selectuser_id,item_id
fromtianchi_lbs.tianchi_mobile_recommend_train_user
wheredatediff(to_date('${label_day}','yyyy-mm-dd hh'),to_date(time,'yyyy-mm-ddhh'),'dd')<3 and datediff(to_date('${label_day}','yyyy-mm-ddhh'),to_date(time,'yyyy-mm-dd hh'),'dd')>0
group byuser_id,item_id
having sum(case whenbehavior_type=4 then 1 else 0 end)=0;
 
--生成分类标签
drop table if existstemp_label_table;
create tabletemp_label_table as
selecta.user_id,a.item_id,case when b.user_id is null and b.item_id is null then 0else 1 end as tag
from
--
temp_pairs_table a
left outer join
(
    --分类标签
    select distinct user_id,item_id
    fromtianchi_lbs.tianchi_mobile_recommend_train_user
    wheredatediff(to_date('${label_day}','yyyy-mm-dd hh'),to_date(time,'yyyy-mm-ddhh'),'dd')=0 and behavior_type=4
) b
ona.user_id=b.user_id and a.item_id=b.item_id;
 
 
--------------------------------------------------------------------------------
--                              数据融合                                     --
--------------------------------------------------------------------------------
 
--融合特征，生成特征总表（没有归一化处理）
drop table if exists${table_label}_feature_table;
create table${table_label}_feature_table as
selecta.user_id,a.item_id,a.tag,
--item特征
i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12,i13,i14,i15,i16,i17,i18,i19,i20,i21,i22,i23,i24
,i25,i26,i27,i28,i29,i30,i31,i32,i33,i34,i35,i36,i37,i38,i39,i40,i41,i42,i43,i44,i45,i46,i47,i48,i49,i50,i51,i52,i53,i54
--ui特征
,ui1,ui2,ui3,ui4,ui5,ui6,ui7,ui8,ui9,ui10,ui12,ui13,ui14,ui15,ui16,ui17,ui18,ui19,ui20,ui21,ui22,ui23,ui24,ui25,ui26,ui27,ui28,ui29,ui30,ui31,ui32,ui33,ui34,ui35
,ui36,ui37,ui38,ui39,ui40,ui41,ui42,ui43,ui44,ui45,ui46,ui47,ui48,ui49,ui50,ui51,ui52,ui53,ui54,ui55
--user特征
,u1,u2,u3,u4,u5,u6,u7,u8,u9,u10,u11,u12,u13,u14,u15,u16,u17,u18,u19,u20,u21,u22
,u23,u24,u25,u26,u27,u28,u29,u30,u31,u32,u33,u34,u35,u36,u37,u38,u39,u40,u41,u42,u43,u44,u45
fromtemp_label_table a
left outer join${table_label}_item_features
ona.item_id=${table_label}_item_features.item_id
left outer join${table_label}_ui_features
ona.user_id=${table_label}_ui_features.user_id anda.item_id=${table_label}_ui_features.item_id
left outer join${table_label}_user_features
ona.user_id=${table_label}_user_features.user_id;
 
--删除中间表
drop table if existstemp_label_table;
drop table if existstemp_pairs_table;
 
--------------------------------------------------------------------------------
--                              归一化                                       --
--------------------------------------------------------------------------------
 
--对特征做归一化处理
drop table if exists${table_label}_normal;
PAI -name Normalize-project algo_public -DkeepOriginal="false"-DoutputTableName="${table_label}_normal"-DinputTableName="${table_label}_feature_table"
-DselectedColNames="i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12,i13,i14,i15,i16,i17,i18,i19,i20,i21,i22,i23,i24,i25,i26,i27,i28,i29,i30,i31,i32,i33,i34,i35,i36,i37,i38,i39,i40,i41,i42,i43,i44,i45,i46,i47,i48,i49,i50,i51,i52,i53,i54,ui1,ui2,ui3,ui4,ui5,ui6,ui7,ui8,ui9,ui10,ui12,ui13,ui14,ui15,ui16,ui17,ui18,ui19,ui20,ui21,ui22,ui23,ui24,ui25,ui26,ui27,ui28,ui29,ui30,ui31,ui32,ui33,ui34,ui35,ui36,ui37,ui38,ui39,ui40,ui41,ui42,ui43,ui44,ui45,ui46,ui47,ui48,ui49,ui50,ui51,ui52,ui53,ui54,ui55,u1,u2,u3,u4,u5,u6,u7,u8,u9,u10,u11,u12,u13,u14,u15,u16,u17,u18,u19,u20,u21,u22,u23,u24,u25,u26,u27,u28,u29,u30,u31,u32,u33,u34,u35,u36,u37,u38,u39,u40,u41,u42,u43,u44,u45";
 
 
--------------------------------------------------------------------------------
--                              欠采样                                       --
--------------------------------------------------------------------------------
 
--只对训练数据集有意义
--拆分数据为正例和反例
drop table if existstemp_pos_table;
create tabletemp_pos_table as
select * from${table_label}_normal where tag=1;
 
drop table if existstemp_nat_table;
create tabletemp_nat_table as
select * from${table_label}_normal where tag=0;
 
--对反例数据做欠采样
drop table if existssub_nat_table;
PAI -nameRandomSample -project algo_public -Dreplace="false"-DoutTableName="sub_nat_table" -DsampleSize="2000000"-DinputTableName="temp_nat_table";
 
--合并采样后的正反例数据，删除零时表
drop table if exists${table_label}_under_sample;
create table${table_label}_under_sample as
select * from
(
    select * from sub_nat_table
    union all
    select * from temp_pos_table
) t;
drop table if existssub_nat_table;
drop table if existstemp_nat_table;
drop table if existstemp_pos_table;
 
--使用lr-gbdt串联分类器，使用lr过滤掉大部分反例样本，然后再使用二级分类器做分类
 
--来个约定先
--lr分类器（一级分类器）：lr_model_0622
--gbdt分类器（二级分类器）:gbdt_model_0622
--
 
-- --清洁工先上场清扫垃圾
-- --清除上次的分类器
DROP OFFLINEMODEL IFEXISTS lr_model_0626;
drop offlinemodel ifexists gbdt_model_0626;
 
-- --训练工作开始啦！！！
-- --切分16日数据为两个部分各50%
drop table if existstrain_p1;
drop table if existstrain_p2;
PAI -nameDeclustering -project algo_public -Dfactors="0.5,0.5"-DoutputTableNames="train_p1,train_p2"-DinputTableName="16_under_sample";
 
--选择训练数据集训练一级分类器（16日train_p1数据）
DROP OFFLINEMODEL IFEXISTS lr_model_0626;
PAI -nameLogisticRegression -project algo_public -DmodelName="lr_model_0626"
-DregularizedLevel="20"-DmaxIter="200" -Depsilon="0.000001"-DlabelColName="tag"
-DfeatureColNames="i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12,i13,i14,i15,i16,i17,i18,i19,i20,i21,i22,i23,i24,i25,i26,i27,i28,i29,i30,i31,i32,i33,i34,i35,i36,i37,i38,i39,i40,i41,i42,i43,i44,i45,i46,i47,i48,i49,i50,i51,i52,i53,i54,ui1,ui2,ui3,ui4,ui5,ui6,ui7,ui8,ui9,ui10,ui12,ui13,ui14,ui15,ui16,ui17,ui18,ui19,ui20,ui21,ui22,ui23,ui24,ui25,ui26,ui27,ui28,ui29,ui30,ui31,ui32,ui33,ui34,ui35,ui36,ui37,ui38,ui39,ui40,ui41,ui42,ui43,ui44,ui45,ui46,ui47,ui48,ui49,ui50,ui51,ui52,ui53,ui54,ui55,u1,u2,u3,u4,u5,u6,u7,u8,u9,u10,u11,u12,u13,u14,u15,u16,u17,u18,u19,u20,u21,u22,u23,u24,u25,u26,u27,u28,u29,u30,u31,u32,u33,u34,u35,u36,u37,u38,u39,u40,u41,u42,u43,u44,u45"
-DgoodValue="1"-DinputTableName="train_p1";
 
--使用一级分类器预测train_p2
drop table if existsp2_after_lr;
PAI -name Prediction-project algo_public -DdetailColName="prediction_detail1"
-DsplitCharacteristic="1"  -DmodelName="lr_model_0626"-DresultColName="prediction_result1"-DoutputTableName="p2_after_lr"-DscoreColName="prediction_score1" -DinputTableName="train_p2"-DlabelValueToPredict="1"
-DappendColNames="user_id,item_id,tag,i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12,i13,i14,i15,i16,i17,i18,i19,i20,i21,i22,i23,i24,i25,i26,i27,i28,i29,i30,i31,i32,i33,i34,i35,i36,i37,i38,i39,i40,i41,i42,i43,i44,i45,i46,i47,i48,i49,i50,i51,i52,i53,i54,ui1,ui2,ui3,ui4,ui5,ui6,ui7,ui8,ui9,ui10,ui12,ui13,ui14,ui15,ui16,ui17,ui18,ui19,ui20,ui21,ui22,ui23,ui24,ui25,ui26,ui27,ui28,ui29,ui30,ui31,ui32,ui33,ui34,ui35,ui36,ui37,ui38,ui39,ui40,ui41,ui42,ui43,ui44,ui45,ui46,ui47,ui48,ui49,ui50,ui51,ui52,ui53,ui54,ui55,u1,u2,u3,u4,u5,u6,u7,u8,u9,u10,u11,u12,u13,u14,u15,u16,u17,u18,u19,u20,u21,u22,u23,u24,u25,u26,u27,u28,u29,u30,u31,u32,u33,u34,u35,u36,u37,u38,u39,u40,u41,u42,u43,u44,u45";
 
 --拆分数据集（正例有2.3，反例有528）
drop table if existstemp_nat;
create tabletemp_nat as
select * fromp2_after_lr where tag=0 and prediction_score1>0.1;
 
drop table if existstemp_pos;
create tabletemp_pos as
select * fromp2_after_lr where tag=1 and prediction_score1>0.1;
 
--对反例做欠采样160000
drop table if existssub_nat;
PAI -nameRandomSample -project algo_public -DoutTableName="sub_nat"
-DsampleSize="160000"-DinputTableName="temp_nat";
 
--对正例做欠采样80000,60000,40000
drop table if existssub_pos;
PAI -nameRandomSample -project algo_public -DoutTableName="sub_pos"
-DsampleSize="65000"-DinputTableName="temp_pos";
 
--构建二级训练数据
drop table if existstrain_2;
create table train_2as
select * from
(
select * fromsub_pos
union all
select * fromsub_nat
)t;
 
--训练二级分类器
drop offlinemodel ifexists gbdt_model_0630;
PAI -name GBDT_LR-project algo_public -DfeatureSplitValueMaxSize="500"-DrandSeed="0"
-Dshrinkage="0.05"-DmaxLeafCount="32" -DlabelColName="tag"-DinputTableName="train_2"
-DminLeafSampleCount="500"-DsampleRatio="0.6" -DmaxDepth="11"-DmodelName="gbdt_model_0630" -DmetricType="2"-DfeatureRatio="0.6" -DtestRatio="0.2"  -DtreeCount="500"
-DfeatureColNames="i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12,i13,i14,i15,i16,i17,i18,i19,i20,i21,i22,i23,i24,i25,i26,i27,i28,i29,i30,i31,i32,i33,i34,i35,i36,i37,i38,i39,i40,i41,i42,i43,i44,i45,i46,i47,i48,i49,i50,i51,i52,i53,i54,ui1,ui2,ui3,ui4,ui5,ui6,ui7,ui8,ui9,ui10,ui12,ui13,ui14,ui15,ui16,ui17,ui18,ui19,ui20,ui21,ui22,ui23,ui24,ui25,ui26,ui27,ui28,ui29,ui30,ui31,ui32,ui33,ui34,ui35,ui36,ui37,ui38,ui39,ui40,ui41,ui42,ui43,ui44,ui45,ui46,ui47,ui48,ui49,ui50,ui51,ui52,ui53,ui54,ui55,u1,u2,u3,u4,u5,u6,u7,u8,u9,u10,u11,u12,u13,u14,u15,u16,u17,u18,u19,u20,u21,u22,u23,u24,u25,u26,u27,u28,u29,u30,u31,u32,u33,u34,u35,u36,u37,u38,u39,u40,u41,u42,u43,u44,u45";
 
--评测的分类器性能
 
--测试数据集使用17日测试数据
drop table if existspre_after_lr;
PAI -name Prediction-project algo_public -DdetailColName="prediction_detail1"
-DsplitCharacteristic="1"  -DmodelName="lr_model_0626"-DresultColName="prediction_result1"-DoutputTableName="pre_after_lr"-DscoreColName="prediction_score1" -DinputTableName="17_normal"-DlabelValueToPredict="1"
-DappendColNames="user_id,item_id,tag,i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12,i13,i14,i15,i16,i17,i18,i19,i20,i21,i22,i23,i24,i25,i26,i27,i28,i29,i30,i31,i32,i33,i34,i35,i36,i37,i38,i39,i40,i41,i42,i43,i44,i45,i46,i47,i48,i49,i50,i51,i52,i53,i54,ui1,ui2,ui3,ui4,ui5,ui6,ui7,ui8,ui9,ui10,ui12,ui13,ui14,ui15,ui16,ui17,ui18,ui19,ui20,ui21,ui22,ui23,ui24,ui25,ui26,ui27,ui28,ui29,ui30,ui31,ui32,ui33,ui34,ui35,ui36,ui37,ui38,ui39,ui40,ui41,ui42,ui43,ui44,ui45,ui46,ui47,ui48,ui49,ui50,ui51,ui52,ui53,ui54,ui55,u1,u2,u3,u4,u5,u6,u7,u8,u9,u10,u11,u12,u13,u14,u15,u16,u17,u18,u19,u20,u21,u22,u23,u24,u25,u26,u27,u28,u29,u30,u31,u32,u33,u34,u35,u36,u37,u38,u39,u40,u41,u42,u43,u44,u45";
 
--使用一级分类器过滤数据
drop table if existspre_for_gbdt;
create tablepre_for_gbdt as
selectuser_id,item_id,tag,i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12,i13,i14,i15,i16,i17,i18,i19,i20,i21,i22,i23,i24,i25,i26,i27,i28,i29,i30,i31,i32,i33,i34,i35,i36,i37,i38,i39,i40,i41,i42,i43,i44,i45,i46,i47,i48,i49,i50,i51,i52,i53,i54,ui1,ui2,ui3,ui4,ui5,ui6,ui7,ui8,ui9,ui10,ui12,ui13,ui14,ui15,ui16,ui17,ui18,ui19,ui20,ui21,ui22,ui23,ui24,ui25,ui26,ui27,ui28,ui29,ui30,ui31,ui32,ui33,ui34,ui35,ui36,ui37,ui38,ui39,ui40,ui41,ui42,ui43,ui44,ui45,ui46,ui47,ui48,ui49,ui50,ui51,ui52,ui53,ui54,ui55,u1,u2,u3,u4,u5,u6,u7,u8,u9,u10,u11,u12,u13,u14,u15,u16,u17,u18,u19,u20,u21,u22,u23,u24,u25,u26,u27,u28,u29,u30,u31,u32,u33,u34,u35,u36,u37,u38,u39,u40,u41,u42,u43,u44,u45
from pre_after_lrwhere prediction_score1>0.1;
 
--使用二级分类器预测
drop table if existspre_after_gbdt;
PAI -name Prediction-project algo_public -DdetailColName="prediction_detail2"-DsplitCharacteristic="1"-DappendColNames="user_id,item_id,tag"
-DmodelName="gbdt_model_0629"-DresultColName="prediction_result2"-DoutputTableName="pre_after_gbdt"
-DscoreColName="prediction_score2"-DinputTableName="pre_for_gbdt" -DlabelValueToPredict="1";
--计算混淆矩阵
drop table if existsgbdtlr_confusion;
PAI -nameconfusionmatrix -project algo_public-DoutputTableName="gbdtlr_confusion" -DlabelColName="tag"-DpredictionColName="prediction_result2"-DinputTableName="pre_after_gbdt";
 --显示混淆矩阵内容
 select * from gbdtlr_confusion;
 
 
--------------------------------------------------------------------------------
--使用18日测试数据
drop table if existspre_after_lr_181;
PAI -name Prediction-project algo_public -DdetailColName="prediction_detail1"
-DsplitCharacteristic="1"  -DmodelName="lr_model_0626"-DresultColName="prediction_result1"-DoutputTableName="pre_after_lr_181"-DscoreColName="prediction_score1" -DinputTableName="18_normal"-DlabelValueToPredict="1"
-DappendColNames="user_id,item_id,tag,i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12,i13,i14,i15,i16,i17,i18,i19,i20,i21,i22,i23,i24,i25,i26,i27,i28,i29,i30,i31,i32,i33,i34,i35,i36,i37,i38,i39,i40,i41,i42,i43,i44,i45,i46,i47,i48,i49,i50,i51,i52,i53,i54,ui1,ui2,ui3,ui4,ui5,ui6,ui7,ui8,ui9,ui10,ui12,ui13,ui14,ui15,ui16,ui17,ui18,ui19,ui20,ui21,ui22,ui23,ui24,ui25,ui26,ui27,ui28,ui29,ui30,ui31,ui32,ui33,ui34,ui35,ui36,ui37,ui38,ui39,ui40,ui41,ui42,ui43,ui44,ui45,ui46,ui47,ui48,ui49,ui50,ui51,ui52,ui53,ui54,ui55,u1,u2,u3,u4,u5,u6,u7,u8,u9,u10,u11,u12,u13,u14,u15,u16,u17,u18,u19,u20,u21,u22,u23,u24,u25,u26,u27,u28,u29,u30,u31,u32,u33,u34,u35,u36,u37,u38,u39,u40,u41,u42,u43,u44,u45";
 
--使用一级分类器过滤数据
drop table if existspre_for_gbdt_181;
create tablepre_for_gbdt_181 as
selectuser_id,item_id,tag,i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12,i13,i14,i15,i16,i17,i18,i19,i20,i21,i22,i23,i24,i25,i26,i27,i28,i29,i30,i31,i32,i33,i34,i35,i36,i37,i38,i39,i40,i41,i42,i43,i44,i45,i46,i47,i48,i49,i50,i51,i52,i53,i54,ui1,ui2,ui3,ui4,ui5,ui6,ui7,ui8,ui9,ui10,ui12,ui13,ui14,ui15,ui16,ui17,ui18,ui19,ui20,ui21,ui22,ui23,ui24,ui25,ui26,ui27,ui28,ui29,ui30,ui31,ui32,ui33,ui34,ui35,ui36,ui37,ui38,ui39,ui40,ui41,ui42,ui43,ui44,ui45,ui46,ui47,ui48,ui49,ui50,ui51,ui52,ui53,ui54,ui55,u1,u2,u3,u4,u5,u6,u7,u8,u9,u10,u11,u12,u13,u14,u15,u16,u17,u18,u19,u20,u21,u22,u23,u24,u25,u26,u27,u28,u29,u30,u31,u32,u33,u34,u35,u36,u37,u38,u39,u40,u41,u42,u43,u44,u45
frompre_after_lr_181 where prediction_score1>0.1;
 
--使用二级分类器预测
drop table if existspre_after_gbdt_181;
PAI -name Prediction-project algo_public -DdetailColName="prediction_detail2"-DsplitCharacteristic="1"-DappendColNames="user_id,item_id,tag"
-DmodelName="gbdt_model_0630"-DresultColName="prediction_result2"-DoutputTableName="pre_after_gbdt_181"
-DscoreColName="prediction_score2"-DinputTableName="pre_for_gbdt_181"-DlabelValueToPredict="1";
 
drop table if existstianchi_mobile_recommendation_predict_0630;
create tabletianchi_mobile_recommendation_predict_0630 as
select distinctb.user_id,b.item_id
from
tianchi_lbs.tianchi_mobile_recommend_train_itema
joinpre_after_gbdt_181 b
ona.item_id=b.item_id and b.prediction_result2="1";
