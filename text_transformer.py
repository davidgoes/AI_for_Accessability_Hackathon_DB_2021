#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 18:57:08 2021

@author: DavidGoes
"""
#Source: https://towardsdatascience.com/simple-abstractive-text-summarization-with-pretrained-t5-text-to-text-transfer-transformer-10f6d602c426

import torch
import json 
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config

model = T5ForConditionalGeneration.from_pretrained('t5-small')
tokenizer = T5Tokenizer.from_pretrained('t5-small')
device = torch.device('cpu')

text ="""
The US has "passed the peak" on new coronavirus cases, President Donald Trump said and predicted that some states would reopen this month.
The US has over 637,000 confirmed Covid-19 cases and over 30,826 deaths, the highest for any country in the world.
At the daily White House coronavirus briefing on Wednesday, Trump said new guidelines to reopen the country would be announced on Thursday after he speaks to governors.
"We'll be the comeback kids, all of us," he said. "We want to get our country back."
The Trump administration has previously fixed May 1 as a possible date to reopen the world's largest economy, but the president said some states may be able to return to normalcy earlier than that.
"""

text_2 = """he US has "passed the peak" on new coronavirus cases, President Donald Trump said and predicted that some states would reopen this month.
The US has over 637,000 confirmed Covid-19 cases and over 30,826 deaths, the highest for any country in the world. It achieves state-of-the-art results on multiple NLP tasks like summarization, question answering, machine translation etc using a text-to-text transformer trained on a large text corpus.
Today we will see how we can use huggingface’s transformers library to summarize any given text. T5 is an abstractive summarization algorithm. It means that it will rewrite sentences when necessary than just picking up sentences directly from the original text"""

text_3 = """It is a long established fact that a reader will be distracted by the readable content of a page when 
looking at its layout. It achieves state-of-the-art results on multiple NLP tasks like summarization, question answering, machine translation etc using a text-to-text transformer trained on a large text corpus. The point of using Lorem Ipsum is that it has a more-or-less normal distribution of letters, as opposed to using 
'Content here, content here', making it look like readable English. Many desktop publishing packages and web page editors now use Lorem Ipsum as their default model text, and a search for 'lorem ipsum' will uncover many web sites still in their infancy. Various """

text_4 = """Thank you for taking the time to meet with our team about the [role title] role at [company name]. It was a pleasure to learn more about your skills and accomplishments.

Unfortunately, our team did not select you for further consideration.

I would like to note that competition for jobs at [company name] is always strong and that we often have to make difficult choices between many high-caliber candidates. Now that we’ve had the chance to know more about you, we will be keeping your resume on f"""

text_5 = """Welcome to the DB Hackathon 2021. Sorry that the invitation E-Mail looks a little bit different than provided before. After accepting your invitation you will be redirected to Microsoft Teams. It can take a few minutes until your access to the Team is established."""

text_6 = """Description
As companies start to build infrastructure using Terraform, the logic required to build self-service automation becomes complex. That's why HCL (HashiCorp Configuration Language) contains expressions that can perform calculations within Terraform configurations. Integrating conditional logic into modules allows them to become dynamic and reusable for different scenarios. 

In this lab, you will create a Storage Account Module that contains conditional logic for deploying a Storage Account. 

Learning Objectives
Upon completion of this lab you will be able to:

Understand how Terraform modules can be made dynamic for reusability
Learn about using conditional expressions within Terraform configurations
Intended Audience
This lab is intended for:

Individuals studying to take the HashiCorp Certified: Terraform Associate exam
Anyone interested in learning how to use Terraform to manage Cloud Service Providers"""

text_7 = """Dear colleagues,

Let’s celebrate! Join us at the Carson Arena for a little bubbly followed by a Mediterranean gourmet buffet on 12 December. After dinner, you can test your bull riding or karaoke skills. If you prefer a quieter atmosphere, you can retreat to one of the cosy lounges and enjoy a drink at the cocktail bar. Look forward to a big surprise in the foyer at 11 p.m.

Kind regards (managing director, company)"""

text_8 = """"Vor der nächsten Runde mit der Kanzlerin und vor der Landtagswahl am 14. März macht sich Baden-Württembergs Ministerpräsident für eine schrittweise Öffnung in den nächsten Wochen stark. Negative Tests sollen zur Eintrittskarte für Geschäfte oder Freizeiteinrichtungen werden. Der baden-württembergische Ministerpräsident Winfried Kretschmann (Grüne) hat für die Ministerpräsidentenkonferenz am nächsten Mittwoch einen ersten Vorschlag für eine Schnell-Teststrategie vorgelegt: Mit Schnelltests, heißt es in dem Papier, das der F.A.Z. vorliegt, müsse es möglich werden, gerade Branchen"""

preprocess_text = text_7.strip().replace("\n","")
t5_prepared_Text = "summarize: "+preprocess_text
print ("original text preprocessed: \n", preprocess_text)

tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)


# summmarize 
summary_ids = model.generate(tokenized_text,
                                    num_beams=4,
                                    no_repeat_ngram_size=2,
                                    min_length=10,
                                    max_length=20,
                                    early_stopping=True)

output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print ("\n\nSummarized text: \n",output, len(output.split(" ")))


# Summarized output from above ::::::::::
# the us has over 637,000 confirmed Covid-19 cases and over 30,826 deaths. 
# president Donald Trump predicts some states will reopen the country in april, he said. 
# "we'll be the comeback kids, all of us," the president says.