from matplotlib import pyplot as plt

def Graph(AUC,HIT,NDCG,MRR,steps,train_loss, batches):
  fig, ax = plt.subplots(1, 5, figsize=(15, 7))
  
  ax[0].plot(steps,AUC,color="red",label="auc")
  ax[0].set_ylabel("AUC")
  ax[0].set_xlabel("steps")
#   ax[0].set_ylim(0.64, 0.70)
  ax[0].legend()
    
  ax[1].plot(steps,MRR,color="red",label="mrr")
  ax[1].set_ylabel("MRR")
  ax[1].set_xlabel("steps")
#   ax[1].set_ylim(0.15, 0.18)
  ax[1].legend()
    
  ax[2].plot(steps,NDCG,color="red",label="ndcg@10")
  ax[2].set_ylabel("ndcg@10")
  ax[2].set_xlabel("steps")
#   ax[2].set_ylim(0.185, 0.215)
  ax[2].legend()

  ax[3].plot(steps,HIT,color="red",label="hit@10")
  ax[3].set_ylabel("HIT@10")
  ax[3].set_xlabel("steps")
#   ax[3].set_ylim(0.38, 0.44)
  ax[3].legend()
    
  ax[4].plot(batches,train_loss,color="red",label="train_loss")
  ax[4].set_ylabel("train_loss")
  ax[4].set_xlabel("batches")
#   ax[3].set_ylim(0.38, 0.44)
  ax[4].legend()
    
    

  plt.show()