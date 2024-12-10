from pydantic import BaseModel

class PropertyPricePred(BaseModel):
    PropertyType: float
    ClubHouse: float 
    School_University_in_Township: float 
    Hospital_in_TownShip: float  
    Mall_in_TownShip: float  
    Park_Jogging_track: float 
    Swimming_Pool: float 
    Gym: float  
    Property_Area_in_Sq_Ft: float  
    Price_by_sub_area: float 
    Noun_Counts: float 
    Verb_Counts: float 
    Adjective_Counts: float 
    Amenities_score: float 
    Price_by_Amenities_score: float 
    bhk_homes: float 
    clubhouse_park: float 
    community_offers: float 
    connectivity_modern: float 
    families_proximity: float 
    gym_perfect: float 
    homes_large: float 
    large_clubhouse: float 
    park_gym: float 
    perfect_families: float 
    properties_offering: float
    proximity_schools: float
    schools_shopping: float
    shopping_malls : float
    spacious_bhk: float