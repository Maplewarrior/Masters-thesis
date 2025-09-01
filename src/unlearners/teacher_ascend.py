import copy
import torch
import numpy as np
import torch.optim as optim
import pdb
from tqdm import tqdm
import math
from torch.utils.data import DataLoader
from src.datasets.synthetic_dataset import SyntheticDataset


"""
"Bad teacher" loss for the maximization step?
    - Turn it into minimizing KL divergence between bad teacher and unlearned model.
"""

class TeacherAscender:
    def __init__(self, model, n_epochs: int, _lambda: float, device: str = 'cpu', MIA: callable = None, js_div_func: callable = None, retrain_model: callable = None, lr: float = 1e-2) -> None:
        self.model = model
        self.n_epochs = n_epochs
        self._lambda = _lambda
        self.device = device
        self.MIA = MIA
        self.lr = lr
        if js_div_func is not None:
            assert retrain_model is not None, "retrain_model must be provided if js_div_func is provided"

        self.js_div_func = js_div_func
        self.retrain_model = retrain_model

    def calculate_FIM(self, dataloader) -> dict:
        """
        A function that calculates the diagonal of the Fisher Information of a model for a specific dataset.

        @param dataloader: An iterable dataloader for which the FIM diagonal should be calculated.
        returns: A dictionary where keys are the names of parameters and values are the FIM of that parameter
        """
        FIM = {k: 0 for k in self.model.state_dict().keys()}
        # define optimizer to allow gradient computation
        optimizer = optim.SGD(self.model.parameters())
        # turn of dropout if applicable
        self.model.eval()

        for i, batch in enumerate(dataloader):
            x = batch[0].to(self.device)
            y = batch[1].to(self.device)
            optimizer.zero_grad()
            # forward pass
            model_out = self.model(x)
            # calculate loss
            loss = self.model.loss(model_out, y)
            # calculate gradients
            loss.backward()
            for i, (name, param) in enumerate(self.model.named_parameters()):
                FIM[name] += param.grad.data.clone().pow(2) #optimizer.param_groups[0]['params'][i].grad.pow(2)
        # account for batched inference
        for k in FIM.keys():
            FIM[k] = FIM[k] / len(dataloader)

        return FIM

    def entropy_schedule(self, num_epochs):
        schedule = [1 / (1+ i**1.75 / num_epochs) for i in range(num_epochs)]
        return schedule

    def calculate_reg_term(self, FIM_original, original_sd):
        reg_term = 0
        for name, param in self.model.named_parameters():
            reg_term += (FIM_original[name] * (param - original_sd[name])**2).sum()
        return reg_term

    def calculate_fit_term(self, model_out, y):
        model_loss = self.model.loss(model_out, y)
        return model_loss

    def calculate_entropy(self, model_out):
        log_probs = (model_out['probabilities'] + 1e-8).log()
        entropy = -(model_out['probabilities'] * log_probs).sum(dim=-1).mean()
        return entropy
    
    def calculate_FIM_ratio(self, FIM_retain, FIM_forget):

        def nanmax(tensor):
            min_value = torch.finfo(tensor.dtype).min
            return tensor.nan_to_num(min_value).max()

        FIM_ratio = {k: 0 for k in FIM_retain.keys()}
        for k in FIM_retain.keys():
            ratio = FIM_retain[k] / FIM_forget[k]
            FIM_ratio[k] = (torch.nan_to_num(ratio, nan=0.0, posinf=nanmax(ratio))).clamp(0., 1e6) / 1000

            ### Addition
            FIM_ratio[k][FIM_ratio[k] == 0] = FIM_ratio[k].mean()

        return FIM_ratio
    

    def mask_FIM_ratio(self, FIM_ratio, FIM_forget):
        """
        Set all the values in FIM_ratio where FIM_forget > 0 to zero
        """
        for k in FIM_ratio.keys():
            FIM_ratio[k][FIM_forget[k] > 0] = 0

        return FIM_ratio

    def eval(self, dataloader):
        self.model.eval()
        losses = []
        acc = 0

        with torch.no_grad():

            for batch in dataloader:
                x = batch[0].to(self.device)
                y = batch[1].to(self.device)

                out = self.model(x)
                loss = self.model.loss(out, y)

                # update metrics
                losses.append(loss.item())
                acc += (out['probabilities'].argmax(dim=1) == y.argmax(dim=1)).sum().item()

        acc = acc / len(dataloader.dataset)
        self.model.train()
        return np.mean(losses), acc

    def construct_validation_set(self, forget_dataloader, val_dataloader):
        """
        A function for constructing a validation set that is "of the same distribution" as the forget dataset. This is the +R step in the method.
        "of the same distribution" is interpreted as being in terms of the label distribution.
        """
        forget_dataset = forget_dataloader.dataset
        val_dataset = val_dataloader.dataset
        X_val = val_dataset.X
        y_val_ohe = val_dataset.y
        y_val = val_dataset.y.argmax(dim=1)
        n_classes = int(y_val.max()+1) # assumes all classes are in the validation set...
        labels, counts = torch.unique(torch.argmax(forget_dataset.y, dim=1), return_counts=True)

        sample_sizes = []

        X_values = []
        y_values = []
        for i, label in enumerate(labels):
            # find val set indexes that correspond to the label
            val_label_idx = torch.where(y_val == label)[0]
            # find sample size
            sample_size = min(counts[i].item(), len(val_label_idx))
            sample_sizes.append(sample_size)
            # draw random samples
            idxs = torch.randperm(len(val_label_idx))[:sample_size]
            # draw subset of data based on index
            X = X_val[val_label_idx[idxs]]
            y = y_val_ohe[val_label_idx[idxs]]
            X_values.append(X)
            y_values.append(y)

        # check if exact distribution could be constructed..
        if not (torch.tensor(sample_sizes) == counts).all():
            print("Exact distribution could not be constructed..")

        X_values = torch.cat(X_values)
        y_values = torch.cat(y_values)

        # set generator for reproducibility
        generator = torch.Generator()
        generator.manual_seed(self.model.seed)
        # Pass torch tensors directly instead of converting to numpy
        dataset = SyntheticDataset(X_values, y_values, n_classes=int(n_classes))
        dataloader = DataLoader(dataset, batch_size=val_dataloader.batch_size)
        return dataloader

    def construct_retain_subset(self, forget_dataloader, retain_dataloader):
        """
        Draws a subset of the retain data with the same label distribution as the forget data
        """
        forget_dataset = forget_dataloader.dataset
        retain_dataset = retain_dataloader.dataset
        X_r = retain_dataset.X
        y_r_ohe = retain_dataset.y
        y_r = retain_dataset.y.argmax(dim=1)
        n_classes = int(y_r.max()+1) # assumes all classes are in the retain set...

        forget_labels, forget_counts = torch.unique(torch.argmax(forget_dataset.y, dim=1), return_counts=True)
        retain_labels, retain_counts = torch.unique(y_r, return_counts=True)

        forget_percentages = forget_counts / forget_counts.sum()

        X_values = []
        y_values = []
        for i, label in enumerate(forget_labels):
            retain_idxs = torch.where(y_r == label)[0]
            sample_size = int(len(retain_idxs) * forget_percentages[i])
            idxs = torch.randperm(len(retain_idxs))[:sample_size]

            X_values.append(X_r[idxs])
            y_values.append(y_r[idxs])

        X_values = torch.cat(X_values)
        y_values = torch.cat(y_values)

        # set generator for reproducibility
        generator = torch.Generator()
        generator.manual_seed(self.model.seed)
        # Pass torch tensors directly instead of converting to numpy
        dataset = SyntheticDataset(X_values, y_values, n_classes=int(n_classes))
        dataloader = DataLoader(dataset, batch_size=retain_dataloader.batch_size)

        return dataloader

    def calculate_classwise_kl(self, out_f, y_f, class_rewind_probs):
        """
        out_f (dict): A model output for the forget batch.
        y_f (tensor): The class labels corresponding to the forget batch.
        class_rewind_probs (tensor): A n_classses x n_classes tensor containing the lass probabilities of the model 
                                     obtained immediately after the min-max phase on the rewind set.
        """
        
    def calculate_entanglement_score(self, retain_loader, forget_loader, stop_idx):
        retain_embeddings = []
        forget_embeddings = []
        for batch in retain_loader:
            x = batch[0].to(self.device)
            emb = self.model.inference(x, stop_idx=stop_idx)['logits']
            retain_embeddings.append(emb)
        
        for batch in forget_loader:
            x = batch[0].to(self.device)
            emb = self.model.inference(x, stop_idx=stop_idx)['logits']
            forget_embeddings.append(emb)
        
        retain_embeddings = torch.cat(retain_embeddings)
        forget_embeddings = torch.cat(forget_embeddings)
        
        mu_retain = retain_embeddings.mean(dim=0)
        mu_forget = forget_embeddings.mean(dim=0)
        mu_total = torch.cat([retain_embeddings, forget_embeddings]).mean(dim=0)

        (mu_retain * (len(retain_loader) / (len(retain_loader) + len(forget_loader))) + mu_forget*(len(forget_loader) / (len(retain_loader) + len(forget_loader))))
        ES_enumerator = (
            (retain_embeddings - mu_retain).norm(dim=-1, p=2).pow(2).mean() +
            (forget_embeddings - mu_forget).norm(dim=-1, p=2).pow(2).mean()
        )
        ES_denominator = 0.5 * (
            (mu_retain - mu_total).norm(p=2) + (mu_forget - mu_total).norm(p=2)
        )
        # pdb.set_trace()
        return ES_enumerator / ES_denominator

    def calculate_kl_div(self, p, q):
        p_log = torch.clamp(p, min=1e-8).log()
        q = torch.clamp(q, min=1e-8) # avoid numeric issues
        return torch.nn.functional.kl_div(p_log, q, reduction='batchmean')

    def collect_avg_classwise_probs(self, dataloader):
        avg_probs = []
        probs = []
        ys = []
        for batch in dataloader:
            x = batch[0].to(self.device)
            y = batch[1].to(self.device)
            
            probs.append(self.model.inference(x)['probabilities'])
            ys.append(y)
        
        probs = torch.cat(probs)
        ys = torch.cat(ys)
        
        for label in ys.unique(dim=0):
            avg_probs.append(probs[torch.all(ys == label, dim=1)].mean(dim=0).unsqueeze(0))

        return torch.cat(avg_probs)

    def __call__(self, retain_loader, forget_loader, val_loader=None, eval: bool = False, verbose: bool = True, version: str = "entropy-retain", is_FIM_ratio: bool = True):
        """
        Performs gradient ascend on forget set labels while regularizing with ∑ F (p_u - p_o)^2

        Options:
            - Maximize entropy of label distribution on forget data --> probably better for OOD points
            - Maximize cross entropy between prediction and y on forget data --> works better for data poisoning.
            - Both?
        """
        original_model = copy.deepcopy(self.model)
        if val_loader is not None:
            validate_err_dataloader = self.construct_validation_set(forget_loader, val_loader)

        if version not in ["ce", "entropy", "ce-retain", "entropy-retain", "ce-retain-no-reg", "entropy-retain-no-reg", "entropy-retain-repair"]:
            raise ValueError(f"Invalid version: {version}, must be one of: ce, entropy, ce-retain, entropy-retain, ce-retain-no-reg, entropy-retain-no-reg, entropy-retain-repair")

        metrics = {'retain': {'acc': []},
                   'forget': {'acc': []},
                   'val': {'acc': []},
                   'rewind': {'acc': []},
                   "loss_terms": {"reg": [], "reg_weighted": [], "ascend": [], "repair": [], 'full': []}
                  }
        if self.MIA is not None:
            metrics['mia'] = []

        if self.js_div_func is not None:
            metrics['js_div'] = {"retain": [], "forget": [], "val": []}

        # calculate FIM for original model on retain set
        original_sd = copy.deepcopy(self.model.state_dict())
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        FIM_forget = self.calculate_FIM(forget_loader) # used for descent
        FIM_original = self.calculate_FIM(retain_loader) # used for ascent
        FIM_ratio = self.calculate_FIM_ratio(FIM_original, FIM_forget)
        FIM_ratio_masked = self.mask_FIM_ratio(FIM_ratio, FIM_forget)

        # Calculate total number of batches for the inner loop
        n_batches = len(forget_loader)

        ### Additions ###
        n_forget_epochs = self.n_epochs // 2
        
        retain_loader_sub = self.construct_retain_subset(forget_loader, retain_loader)
        
        ent_schedule = self.entropy_schedule(self.n_epochs // 2)
        ent_repair_schedule = list(reversed(self.entropy_schedule(self.n_epochs // 2)))
        
        total_steps = n_batches * self.n_epochs
        warmup_steps = int(0.15 * self.n_epochs)

        def lr_schedule(current_step):
            if current_step < warmup_steps:
                # Linear warm-up
                return float(current_step) / float(max(1, warmup_steps))
            else:
                # Cosine annealing decay
                progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_schedule)
        
        probs_rewind = []
        for b in validate_err_dataloader:
            x_v = b[0].to(self.device)
            probs_rewind.append(self.model.inference(x_v)['probabilities'])
        probs_rewind = torch.cat(probs_rewind)
        
        ent_rewind_0 = self.calculate_entropy({'probabilities': probs_rewind})
        ent_repair = ent_rewind_0

        # probs_rewind = []
        # for b in retain_loader:
        #     x_v = b[0].to(self.device)
        #     probs_rewind.append(self.model.inference(x_v)['probabilities'])
        # probs_rewind = torch.cat(probs_rewind)
        
        # ent_retain = self.calculate_entropy({'probabilities': probs_rewind})

        # ent_factor = ent_rewind / ent_retain

        epoch_iterator = tqdm(range(self.n_epochs), desc='Epochs', leave=True) if verbose else range(self.n_epochs)

        for epoch in epoch_iterator:
            if epoch == n_forget_epochs:
                FIM_forget = self.calculate_FIM(forget_loader) # used for descend
                FIM_original = self.calculate_FIM(retain_loader) # used for ascend
                FIM_ratio = self.calculate_FIM_ratio(FIM_original, FIM_forget)

            if self.MIA is not None:
                mia_prob = self.MIA(self.model, retain_loader, forget_loader, val_loader)
                metrics['mia'].append(mia_prob)


            if self.js_div_func is not None:
                # JS_divergence(self, probs_u, probs_c, log_base: float = 2.0)
                probs_u = self.model.inference(retain_loader.dataset.X.to(self.device))['probabilities']
                probs_c = self.retrain_model.inference(retain_loader.dataset.X.to(self.device))['probabilities']
                js_div = self.js_div_func(probs_u, probs_c)
                metrics['js_div']['retain'].append(js_div)
                probs_u = self.model.inference(forget_loader.dataset.X.to(self.device))['probabilities']
                probs_c = self.retrain_model.inference(forget_loader.dataset.X.to(self.device))['probabilities']
                js_div = self.js_div_func(probs_u, probs_c)
                metrics['js_div']['forget'].append(js_div)
                probs_u = self.model.inference(val_loader.dataset.X.to(self.device))['probabilities']
                probs_c = self.retrain_model.inference(val_loader.dataset.X.to(self.device))['probabilities']
                js_div = self.js_div_func(probs_u, probs_c)
                metrics['js_div']['val'].append(js_div)

            if eval:
                _, forget_acc = self.eval(forget_loader)
                metrics['forget']['acc'].append(forget_acc)
                _, retain_acc = self.eval(retain_loader)
                metrics['retain']['acc'].append(retain_acc)
                _, val_acc = self.eval(val_loader)
                metrics['val']['acc'].append(val_acc)
                _, validate_err_acc = self.eval(validate_err_dataloader)
                metrics['rewind']['acc'].append(validate_err_acc)

            #### Gradient ascent
            batch_iterator = tqdm(enumerate(forget_loader),
                                desc=f'Batch Processing',
                                total=n_batches,
                                leave=False) if verbose else enumerate(forget_loader)
            
            if epoch == 0 or (n_forget_epochs - epoch) == 1 or (epoch == (self.n_epochs - 1)):
                probs_rewind = []
                for b in retain_loader_sub:
                    x_v = b[0].to(self.device)
                    probs_rewind.append(self.model.inference(x_v)['probabilities'])
                probs_rewind = torch.cat(probs_rewind)
                
                ent_rs = self.calculate_entropy({'probabilities': probs_rewind})

                probs_rewind = []
                for b in validate_err_dataloader:
                    x_v = b[0].to(self.device)
                    probs_rewind.append(self.model.inference(x_v)['probabilities'])
                probs_rewind = torch.cat(probs_rewind)
                
                ent_rewind = self.calculate_entropy({'probabilities': probs_rewind})

                probs_rewind = []
                for b in retain_loader:
                    x_v = b[0].to(self.device)
                    probs_rewind.append(self.model.inference(x_v)['probabilities'])
                probs_rewind = torch.cat(probs_rewind)
                
                ent_retain = self.calculate_entropy({'probabilities': probs_rewind})

                probs_rewind = []
                for b in forget_loader:
                    x_v = b[0].to(self.device)
                    probs_rewind.append(self.model.inference(x_v)['probabilities'])
                probs_rewind = torch.cat(probs_rewind)

                ent_forget = self.calculate_entropy({'probabilities': probs_rewind})
                print(f'Ent forget: {ent_forget}')
                print(f'Ent retain: {ent_retain}')
                print(f'Ent retain_sub: {ent_rs}')
                print(f'Ent rewind: {ent_rewind}')
                # pdb.set_trace()
                #ent_rewind = 0.5 * ent_rewind + 0.5 * ent_retain


            for batch_idx, batch in batch_iterator:
                optimizer.zero_grad()

                x_f = batch[0].to(self.device)
                y_f = batch[1].to(self.device)

                batch_retain = next(iter(retain_loader))
                x_r = batch_retain[0].to(self.device)
                y_r = batch_retain[1].to(self.device)

                batch_retain_sub = next(iter(retain_loader_sub))
                x_r_sub = batch_retain_sub[0].to(self.device)
                y_r_sub = batch_retain_sub[1].to(self.device)

                out_f = self.model(x_f)
                out_r = self.model(x_r)
                out_r_sub = self.model(x_r_sub)

                reg_term = self.calculate_reg_term(FIM_ratio, original_sd) if is_FIM_ratio else self.calculate_reg_term(FIM_original, original_sd)
                weighted_reg_term = self._lambda / 2 * reg_term

                if version == "ce":
                    forget_term = self.calculate_fit_term(out_f, y_f)
                    loss = -forget_term + weighted_reg_term # maximize CE, minimize reg term

                elif version == "entropy":
                    forget_term = self.calculate_entropy(out_f)
                    loss = -forget_term + weighted_reg_term # maximize entropy, minimize reg term

                elif version == "ce-retain":
                    forget_term = self.calculate_fit_term(out_f, y_f)
                    retain_term = self.calculate_fit_term(out_r, y_r)
                    loss = -forget_term + retain_term + weighted_reg_term # maximize CE, minimize reg term
                    metrics['loss_terms']['repair'].append(retain_term.item())

                elif version == "entropy-retain":
                    forget_term = self.calculate_entropy(out_f)
                    retain_term = self.calculate_fit_term(out_r, y_r)
                    loss = -forget_term + retain_term + weighted_reg_term # maximize entropy, minimize reg term
                    metrics['loss_terms']['repair'].append(retain_term.item())

                elif version == "entropy-retain-repair":
                    # elif version == 'l1-reg':
                    if epoch < (n_forget_epochs):
                        # forget_term = self.calculate_entropy(out_f) * ent_schedule[epoch]
                        # pdb.set_trace()
                        out_r_original = original_model.inference(x_r)
                        forget_term = self.calculate_entropy(out_f) * ent_schedule[epoch]
                        retain_term = self.calculate_kl_div(out_r['probabilities'], out_r_original['probabilities']) # self.calculate_fit_term(out_r, y_r) +
                        # entropy_retain_sub = self.calculate_entropy(out_r_sub)
                        # retain_term_sub = self.calculate_fit_term(out_r_sub, y_r_sub)
                        loss = -forget_term + retain_term + weighted_reg_term # maximize entropy, minimize fit, reg term
                    else:
                        # reg_term = #self._lambda / 2 * self.calculate_reg_term(FIM_ratio_masked, original_sd)
                        # retain_term = self.calculate_fit_term(out_r, y_r)
                        out_r_original = original_model.inference(x_r)
                        retain_term = self.calculate_fit_term(out_r, y_r) + self.calculate_kl_div(out_r['probabilities'], out_r_original['probabilities'])
                        # forget_term = self.calculate_entropy(out_f)

                        loss = retain_term + weighted_reg_term #+ ent_repair_schedule[epoch - n_forget_epochs] * (forget_term - ent_repair)**2

                    # # retain_term = self.calculate_fit_term(out_r, y_r)
                    # # loss = -ent_schedule[epoch]*forget_term + retain_term + weighted_reg_term # maximize entropy, minimize fit, reg term
                    # metrics['loss_terms']['repair'].append(retain_term.item())
                
                elif version == 'l1-reg':
                    retain_term = self.calculate_fit_term(out_r, y_r)
                    l1 = torch.norm(torch.cat([p.view(-1) for p in self.model.parameters()]), 1)
                    loss = retain_term + weighted_reg_term + l1


                elif version == "entropy-retain-no-reg":
                    forget_term = self.calculate_entropy(out_f)
                    retain_term = self.calculate_fit_term(out_r, y_r)
                    loss = -forget_term + retain_term # maximize entropy, minimize reg term
                    metrics['loss_terms']['repair'].append(retain_term.item())

                elif version == "ce-retain-no-reg":
                    forget_term = self.calculate_fit_term(out_f, y_f)
                    retain_term = self.calculate_fit_term(out_r, y_r)
                    loss = -forget_term + retain_term # minimize CE, maximize reg term
                    metrics['loss_terms']['repair'].append(retain_term.item())
                

                # append loss to metrics
                metrics['loss_terms']['full'].append(loss.item())
                metrics['loss_terms']['reg_weighted'].append(weighted_reg_term.item())
                metrics['loss_terms']['reg'].append(reg_term.item())
                metrics['loss_terms']['ascend'].append(forget_term.item())

                loss.backward()
                optimizer.step()
                if lr_scheduler is not None:
                    lr_scheduler.step()

        if eval:
            return metrics

