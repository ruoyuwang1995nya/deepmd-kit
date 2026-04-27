* Install the latest `property-derv` branch.

* Run `convert_to_deepmd.py` to convert the raw data into deepmd format.

* Run `dp --pt train multi_task_ft.json --finetune DPA2_medium_28_10M_rc0.pt` to train the model using multi-task finetuning. You can download the base model from https://www.aissquare.com/models/detail?pageType=models&name=DPA-2.3.1-v3.0.0rc0&id=287

* After training the model, atomic simulation can be performed using the `ase` package. An example script is provided for at `nvt_custom.py`