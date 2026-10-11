from pathlib import Path
import pandas as pd


ART=Path(
"experiments/07_v6_multidomain/artifacts"
)


rows=[]


configs={
"deepbeat":
{
"consecutive":5,
"positive":1,
"uncertainty":0.05
},

"pulsewatch":
{
"consecutive":3,
"positive":1,
"uncertainty":0.05
},

"liu":
{
"consecutive":1,
"positive":1,
"uncertainty":0.05
}
}


df=pd.read_csv(
ART/"v6h_episode_strict_results.csv"
)


for dataset,cfg in configs.items():

    x=df[
        (df.dataset==dataset)
        &
        (df.consecutive==cfg["consecutive"])
        &
        (df.positive==cfg["positive"])
        &
        (df.uncertainty==cfg["uncertainty"])
    ]

    rows.append(
        x.iloc[0]
    )


out=pd.DataFrame(rows)

out.to_csv(
ART/"v6h_selected_operating_points.csv",
index=False
)

print(out)

