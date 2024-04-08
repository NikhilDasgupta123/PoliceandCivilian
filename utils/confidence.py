import torch


# Detection result
def conFidence(results):
    
    for r in results:
        try:
            if r.boxes.conf[0] > torch.tensor([0.8]).cuda() :
                print(r.boxes.cls.names)
                break
            elif ( torch.tensor([0.]).cuda() in r.boxes.cls ) and ( torch.tensor([1.]).cuda() in r.boxes.cls ) :
                print(r.boxes.cls.names)
                break
            # predicted, but no match confidece or class name
            else :
                #print("\nnothing\n")
                print(r.boxes.cls.names)
                pass
        except :  # no predicted
            # for j in r.boxes.names.dict:
            #     print(j)
            break
      