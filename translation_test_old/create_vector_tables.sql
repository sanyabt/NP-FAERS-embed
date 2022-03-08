CREATE EXTENSION vector;

drop table if exists scratch_embedding.unmap_vector 

CREATE TABLE scratch_embedding.unmap_vector (
seq serial primary key,
drug_name_original varchar,
drug_vector vector(256)
);

drop table if exists scratch_embedding.lookup_vector  

CREATE TABLE scratch_embedding.lookup_vector (
seq serial primary key,
lookup_value varchar,
lookup_vector vector(256)
);

--testing with NP and drug cosine distance values 
--NPs included - Soy, senna, cumin, fig, sage, thyme, Apricot, Black pepper, Celandine, Grapevine, Licorice	
--Suan Cheng, Goldenseal, Tetterwort, Nimtree, Black snakeroot

--is this in order of cosine distance value?
select lookup_value
FROM scratch_embedding.lookup_vector
ORDER BY lookup_vector.lookup_vector <=> 
(select unmap_vector.drug_vector from scratch_embedding.unmap_vector where drug_name_original = 'GOLDENSEAL HERBS')
LIMIT 20

--
with np_str as (
select regexp_replace(c.concept_name,'\[.*','') cn
from staging_vocabulary.concept c
where c.concept_class_id = 'White pepper'
), match_sub as (
select cos_sim_all.*
from scratch_embedding.cos_sim_all inner join np_str on cos_sim_all.lookup_value = upper(np_str.cn)
), match_sub_vecs as (
select match_sub.lookup_value, lookup_vector.lookup_vector,
unmap_vector.drug_name_original, unmap_vector.drug_vector
from scratch_embedding.lookup_vector inner join match_sub on lookup_vector.lookup_value = match_sub.lookup_value
inner join scratch_embedding.unmap_vector on match_sub.drug_name_original = unmap_vector.drug_name_original
), cossim_calc as (
select lookup_value, drug_name_original, cosine_distance(match_sub_vecs.lookup_vector, match_sub_vecs.drug_vector) cossim
from match_sub_vecs
)
select distinct *
from cossim_calc
where cossim >= 0.9
order by lookup_value, cossim desc
;

--
with np_str as (
select regexp_replace(c.concept_name,'\[.*','') cn
from staging_vocabulary.concept c
where c.concept_class_id = 'Licorice'
), match_sub as (
select cos_sim_all.*
from scratch_embedding.cos_sim_all inner join np_str on cos_sim_all.lookup_value = upper(np_str.cn)
), match_sub_vecs as (
select match_sub.lookup_value, lookup_vector.lookup_vector,
unmap_vector.drug_name_original, unmap_vector.drug_vector
from scratch_embedding.lookup_vector inner join match_sub on lookup_vector.lookup_value = match_sub.lookup_value
inner join scratch_embedding.unmap_vector on match_sub.drug_name_original = unmap_vector.drug_name_original
), cossim_calc as (
select lookup_value, drug_name_original, cosine_distance(match_sub_vecs.lookup_vector, match_sub_vecs.drug_vector) cossim
from match_sub_vecs
)
select distinct *
from cossim_calc
where cossim >= 0.7
order by lookup_value, cossim desc
;

with match_sub as (
select cos_sim_all.*
from scratch_embedding.cos_sim_all where cos_sim_all.lookup_value = 'GOLDENSEAL'
), match_sub_vecs as (
select match_sub.lookup_value, lookup_vector.lookup_vector,
unmap_vector.drug_name_original, unmap_vector.drug_vector
from scratch_embedding.lookup_vector inner join match_sub on lookup_vector.lookup_value = match_sub.lookup_value
inner join scratch_embedding.unmap_vector on match_sub.drug_name_original = unmap_vector.drug_name_original
), cossim_calc as (
select lookup_value, drug_name_original, cosine_distance(match_sub_vecs.lookup_vector, match_sub_vecs.drug_vector) cossim
from match_sub_vecs
)
select distinct *
from cossim_calc
where cossim >= 0.7
order by lookup_value, cossim desc
;


